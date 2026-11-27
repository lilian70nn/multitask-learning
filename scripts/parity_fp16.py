#!/usr/bin/env python3
"""Check that the fp16 vision stack agrees with fp32.

The stack is loaded twice through server.get_vlm_model(), once with
VISION_DTYPE=fp32 and once with VISION_DTYPE=fp16, each in its own subprocess so
only one copy is in memory at a time. Per segmentation domain, the criteria are:

  * top-1 class agrees in at least 95% of the images (19 of 20),
  * the largest difference of any class probability is below 0.02,
  * the Dice score between the fp32 and fp16 masks (polyp probability >= 0.5)
    is at least 0.98 for every image (two empty masks count as 1.0).

Captions are printed side by side for a manual check (not a pass criterion).

Usage (GPU needed; stop the running service first if memory is tight):

  .venv/bin/python scripts/parity_fp16.py --images-dir /path/to/images
  .venv/bin/python scripts/parity_fp16.py --images a.jpg b.jpg c.jpg

Without --images and --images-dir, the directory is taken from the environment
variable TEST_IMAGES_DIR.

--self-test checks the comparison logic on synthetic data (no GPU, no weights).
Exit code: 0 = passed, 1 = a criterion failed, 2 = could not run.
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import textwrap

import numpy as np

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_IMAGES_DIR = os.environ.get("TEST_IMAGES_DIR", "")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

MIN_TOP1_FRACTION = 0.95
MAX_PROB_DIFF = 0.02
MIN_DICE = 0.98
MASK_THRESH = 0.5


# --------------------------------------------------------------------- comparison
def dice(a, b):
    a, b = a.astype(bool), b.astype(bool)
    total = int(a.sum()) + int(b.sum())
    return 1.0 if total == 0 else 2.0 * int((a & b).sum()) / total


def compare(ref, test, names, class_names):
    """Compare two {probs, masks, polyp, captions} dicts of one domain. Returns True if passed."""
    n = len(names)
    p32, p16 = ref["probs"], test["probs"]
    finite = bool(np.isfinite(p16).all() and np.isfinite(test["polyp"]).all())
    top32, top16 = p32.argmax(axis=1), p16.argmax(axis=1)
    agree = int((top32 == top16).sum())
    needed = math.ceil(MIN_TOP1_FRACTION * n - 1e-9)
    diffs = np.abs(p32.astype(np.float64) - p16.astype(np.float64))
    max_diff = float(diffs.max())
    dices = [dice(ref["masks"][i], test["masks"][i]) for i in range(n)]

    print(f"{'#':>3}  {'image':<30} {'top-1 fp32':<27} {'top-1 fp16':<27} {'max dP':>8} "
          f"{'Dice':>7} {'px fp32':>8} {'px fp16':>8}")
    for i, name in enumerate(names):
        flag = "" if top32[i] == top16[i] and diffs[i].max() < MAX_PROB_DIFF and dices[i] >= MIN_DICE else "  <--"
        print(f"{i + 1:>3}  {name[:30]:<30} "
              f"{class_names[top32[i]][:20]:<20} {p32[i, top32[i]]:.4f} "
              f"{class_names[top16[i]][:20]:<20} {p16[i, top16[i]]:.4f} "
              f"{diffs[i].max():>8.5f} {dices[i]:>7.4f} {int(ref['masks'][i].sum()):>8} "
              f"{int(test['masks'][i].sum()):>8}{flag}")

    print("\nCaptions (fp32 | fp16):")
    same_captions = 0
    for i, name in enumerate(names):
        a, b = str(ref["captions"][i]), str(test["captions"][i])
        same_captions += a == b
        print(f"  [{i + 1}] {name}{'' if a == b else '   (differs)'}")
        left, right = textwrap.wrap(a, 58) or ["-"], textwrap.wrap(b, 58) or ["-"]
        for row in range(max(len(left), len(right))):
            print(f"      {left[row] if row < len(left) else '':<58} | {right[row] if row < len(right) else ''}")
    print(f"  identical captions: {same_captions}/{n}")

    checks = [
        ("fp16 outputs are finite", finite, "no NaN/Inf" if finite else "NaN or Inf found"),
        ("top-1 agreement", agree >= needed, f"{agree}/{n} (need >= {needed})"),
        ("max class probability difference", max_diff < MAX_PROB_DIFF, f"{max_diff:.5f} (need < {MAX_PROB_DIFF})"),
        ("minimum mask Dice", min(dices) >= MIN_DICE,
         f"{min(dices):.4f}, mean {float(np.mean(dices)):.4f} (need >= {MIN_DICE})"),
    ]
    print()
    for label, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")
    return all(ok for _, ok, _ in checks)


# ------------------------------------------------------------------------- worker
def run_worker(dtype, job_path, out_path):
    """Load the stack with VISION_DTYPE=dtype (set by the parent) and dump its outputs."""
    with open(job_path, encoding="utf-8") as handle:
        job = json.load(handle)
    sys.path.insert(0, job["repo"])
    os.chdir(job["repo"])

    import torch
    from PIL import Image
    import server

    if not torch.cuda.is_available():
        print("ERROR: no CUDA GPU visible.", file=sys.stderr)
        return 2
    wanted = torch.float16 if dtype == "fp16" else torch.float32
    arrays = {"class_names": np.array(server.CLASS_NAMES)}

    for domain in job["domains"]:
        model = server.get_vlm_model(domain)
        # Without this guard a loader that ignores VISION_DTYPE would pass trivially.
        for part in ("vision_encoder", "cls_model", "seg_model", "decoder"):
            found = {p.dtype for p in getattr(model, part).parameters() if p.is_floating_point()}
            if found != {wanted}:
                print(f"ERROR: VISION_DTYPE={dtype} but {part} ({domain}) has dtypes {found}.", file=sys.stderr)
                return 2
        probs, masks, polyp, captions = [], [], [], []
        with torch.inference_mode():
            for path in job["images"]:
                image = Image.open(path).convert("RGB")
                # Same calls as /api/analyze: a float32 input that the server code casts itself.
                tensor = server.TRANSFORM(image).unsqueeze(0).to(server.DEVICE)
                result = server.analyze_single_tensor(model, tensor, image.resize((224, 224)), MASK_THRESH)
                # All 16 probabilities, computed as analyze_single_tensor does (EDL, in float32).
                alpha = torch.nn.functional.softplus(model.cls_model(tensor.to(dtype=wanted)).float()) + 1.0
                probs.append((alpha / alpha.sum(dim=-1, keepdim=True))[0].cpu().numpy())
                polyp.append(np.asarray(result["polyp_prob"], dtype=np.float32))
                masks.append(polyp[-1] >= MASK_THRESH)
                captions.append(model.generate(tensor, max_new_tokens=64)[0].strip())
        arrays[f"{domain}.probs"] = np.stack(probs)
        arrays[f"{domain}.polyp"] = np.stack(polyp)
        arrays[f"{domain}.masks"] = np.stack(masks)
        arrays[f"{domain}.captions"] = np.array(captions)
        print(f"  {dtype} / {domain}: {len(job['images'])} images, "
              f"peak VRAM {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB", flush=True)
        del model  # get_vlm_model() moves the previous domain off the GPU itself

    np.savez(out_path, **arrays)
    return 0


# --------------------------------------------------------------------------- main
def select_images(args):
    if args.images:
        paths = args.images
    else:
        directory = args.images_dir or DEFAULT_IMAGES_DIR
        if not directory:
            print("ERROR: pass --images-dir/--images or set TEST_IMAGES_DIR (endoscopy frames).", file=sys.stderr)
            return None
        if not os.path.isdir(directory):
            source = "--images-dir" if args.images_dir else "TEST_IMAGES_DIR"
            print(f"ERROR: {source} is not a directory: {directory}", file=sys.stderr)
            return None
        names = sorted(n for n in os.listdir(directory) if n.lower().endswith(IMAGE_EXTS))
        picked = random.Random(args.seed).sample(names, min(args.num_images, len(names)))
        paths = [os.path.join(directory, n) for n in sorted(picked)]
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing or not paths:
        print(f"ERROR: missing images: {missing or 'none found'}", file=sys.stderr)
        return None
    return [os.path.abspath(p) for p in paths]


def self_test():
    rng = np.random.default_rng(0)
    n, class_names = 20, [f"class-{i}" for i in range(16)]
    probs = rng.dirichlet(np.ones(16) * 0.3, size=n).astype(np.float32)
    polyp = np.zeros((n, 32, 32), np.float32)
    polyp[:-1, 4:20, 4:20] = 0.9  # the last mask stays empty
    ref = {"probs": probs, "polyp": polyp, "masks": polyp >= 0.5, "captions": np.array(["a polyp"] * n)}
    names = [f"img{i}.jpg" for i in range(n)]
    clone = lambda: {k: v.copy() for k, v in ref.items()}

    cases = [("identical", clone(), True)]
    swapped = clone()
    for i in range(2):  # two flipped top-1 classes: 18/20 must fail
        order = np.argsort(swapped["probs"][i])
        swapped["probs"][i, order[-1]], swapped["probs"][i, order[-2]] = probs[i, order[-2]], probs[i, order[-1]]
    cases.append(("two top-1 flips", swapped, False))
    shifted = clone()
    shifted["probs"][0, 0] += 0.03
    cases.append(("probability shift 0.03", shifted, False))
    shrunk = clone()
    shrunk["masks"][0, 4:6, 4:20] = False
    cases.append(("mask mismatch", shrunk, False))
    nan = clone()
    nan["polyp"][0, 0, 0] = np.nan
    cases.append(("NaN output", nan, False))

    failures = []
    for label, test, expected in cases:
        print(f"\n=== self-test: {label} (expected {'PASS' if expected else 'FAIL'}) ===")
        if compare(ref, test, names, class_names) != expected:
            failures.append(label)
    print("\nSELF-TEST " + (f"FAILED: {failures}" if failures else "PASSED"))
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", help="directory with endoscopy images (default: $TEST_IMAGES_DIR)")
    parser.add_argument("--images", nargs="+", help="explicit image files (instead of --images-dir)")
    parser.add_argument("--num-images", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0, help="seed for picking images from the directory")
    parser.add_argument("--domains", nargs="+", default=["colonbench", "polyp"], choices=["colonbench", "polyp"])
    parser.add_argument("--repo", default=REPO_DIR, help="repo root containing server.py (default: %(default)s)")
    parser.add_argument("--self-test", action="store_true", help="test the comparison logic on synthetic data")
    parser.add_argument("--worker", choices=["fp32", "fp16"], help=argparse.SUPPRESS)
    parser.add_argument("--job", help=argparse.SUPPRESS)
    parser.add_argument("--out", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        return run_worker(args.worker, args.job, args.out)
    if args.self_test:
        return self_test()

    images = select_images(args)
    if images is None:
        return 2
    if len(images) < 20:
        print(f"WARNING: only {len(images)} images; the plan's criterion is defined on 20.")

    with tempfile.TemporaryDirectory(prefix="endovision_parity_") as tmp:
        job_path = os.path.join(tmp, "job.json")
        with open(job_path, "w", encoding="utf-8") as handle:
            json.dump({"repo": os.path.abspath(args.repo), "images": images, "domains": args.domains}, handle)
        outputs = {}
        for dtype in ("fp32", "fp16"):
            out_path = os.path.join(tmp, f"{dtype}.npz")
            # Process environment takes priority over .env. No web server is started.
            env = dict(os.environ, VISION_DTYPE=dtype, ALLOW_NO_AUTH="1", ENABLE_SYNTHESIS="0")
            print(f"Running {dtype} ...", flush=True)
            code = subprocess.call([sys.executable, os.path.abspath(__file__), "--worker", dtype,
                                    "--job", job_path, "--out", out_path], env=env)
            if code != 0 or not os.path.isfile(out_path):
                print(f"ERROR: the {dtype} run failed (exit code {code}).", file=sys.stderr)
                return 2
            with np.load(out_path, allow_pickle=False) as data:
                outputs[dtype] = {k: data[k] for k in data.files}

    names = [os.path.basename(p) for p in images]
    class_names = [str(c) for c in outputs["fp32"]["class_names"]]
    passed = True
    for domain in args.domains:
        print(f"\n########## segmentation domain: {domain} ##########\n")
        pick = lambda d: {k: outputs[d][f"{domain}.{k}"] for k in ("probs", "polyp", "masks", "captions")}
        passed &= compare(pick("fp32"), pick("fp16"), names, class_names)

    print("\nPARITY " + ("PASSED" if passed else "FAILED"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
