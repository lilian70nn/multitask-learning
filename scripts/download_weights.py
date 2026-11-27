#!/usr/bin/env python3
"""Download only the weight files that server.py loads (about 5.5 GB).

    python scripts/download_weights.py                  # into $WEIGHTS_DIR or ./model_weights
    python scripts/download_weights.py --dest /path
    python scripts/download_weights.py --check-only     # verify an existing folder

The Swin-tiny backbone config and the chat model live in the Hugging Face cache
and are not handled here.
"""
import argparse
import os
import shutil
import sys

REPO_ID = "Lian70/model_weights"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_CLS = "classification/vit_gastronet_dino_edl/hierarchical"
_DOMAINS = ("colonbench", "polyp")
_DECODER_FILES = (
    "added_tokens.json", "config.json", "generation_config.json", "merges.txt", "model.safetensors",
    "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
)

EXPECTED_FILES = (
    ["dino-weights/ema_backbone.pt", f"{_CLS}/groups.json", f"{_CLS}/coarse_best_model.pt"]
    + [f"{_CLS}/swin_group_{i}_best_model.pt" for i in range(3)]
    + [f"segmentation/vit_gastronet_dino_edl_{d}/best_model.pt" for d in _DOMAINS]
    + [f"decoder/dino_gastronet_hierarchical_{d}/baseline_3/adapter.pth" for d in _DOMAINS]
    + [f"decoder/dino_gastronet_hierarchical_{d}/baseline_3/decoder/{name}"
       for d in _DOMAINS for name in _DECODER_FILES]
)


def default_dest() -> str:
    """Same resolution as server.py: WEIGHTS_DIR (environment or .env), else ./model_weights."""
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(REPO_ROOT, ".env"), override=False)
    except ImportError:
        pass
    value = os.environ.get("WEIGHTS_DIR", "").strip() or "model_weights"
    return os.path.abspath(os.path.join(REPO_ROOT, os.path.expanduser(value)))


def flatten(dest: str) -> None:
    """Move files out of the nested model_weights/ folder that the repository may contain."""
    nested = os.path.join(dest, "model_weights")
    if not os.path.isdir(nested):
        return
    for folder, _, files in os.walk(nested):
        target_dir = os.path.join(dest, os.path.relpath(folder, nested))
        os.makedirs(target_dir, exist_ok=True)
        for name in files:
            os.replace(os.path.join(folder, name), os.path.join(target_dir, name))
    shutil.rmtree(nested)


def missing_files(dest: str):
    paths = ((rel, os.path.join(dest, rel)) for rel in EXPECTED_FILES)
    return [rel for rel, path in paths if not (os.path.isfile(path) and os.path.getsize(path) > 0)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", help="target folder (default: $WEIGHTS_DIR or ./model_weights)")
    parser.add_argument("--check-only", action="store_true", help="only verify that all files exist")
    args = parser.parse_args()
    dest = os.path.abspath(os.path.expanduser(args.dest)) if args.dest else default_dest()

    if not args.check_only:
        flatten(dest)
        if missing_files(dest):
            from huggingface_hub import snapshot_download
            # The repository may keep a nested model_weights/ folder: allow both layouts.
            patterns = EXPECTED_FILES + ["model_weights/" + rel for rel in EXPECTED_FILES]
            snapshot_download(REPO_ID, local_dir=dest, allow_patterns=patterns)
            flatten(dest)

    missing = missing_files(dest)
    if missing:
        print(f"{len(missing)} expected file(s) missing in {dest}:", file=sys.stderr)
        for rel in missing:
            print(f"  {rel}", file=sys.stderr)
        return 1
    print(f"All {len(EXPECTED_FILES)} weight files are present in {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
