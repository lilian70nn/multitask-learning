import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from segmentation_vit import hyperkvasir_dataloader as hyperkvasir
from segmentation_vit import segmentation_dataloader as colonbench
from segmentation_vit.train_segmentation import initialize_vit_unet, train_segmentation, compute_metrics


DATASETS = {"hyperkvasir": hyperkvasir, "colonbench": colonbench}
METRICS = ["val_dice", "val_iou", "mean_entropy", "mean_foreground_entropy"]


def sample_seeds(count):
    rng = random.Random(2026)
    seeds, used = [], set()
    while len(seeds) < count:
        seed = rng.randrange(2**31)
        if seed not in used:
            seeds.append(seed)
            used.add(seed)
    return seeds


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def write_csv(frame, path):
    temp = path.with_suffix(".csv.tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def load_or_create_split(args):
    root = args.data_dir
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    split_path = args.output_dir.parent / "dataset_split.json"
    if split_path.is_file():
        split = json.loads(split_path.read_text())
        if split.get("dataset") != args.dataset:
            raise ValueError(f"Dataset name or split format does not match: {split_path}")
        print(f"Using saved split: {split_path}", flush=True)
        return split

    # Reuse the original 85/15 split, including video-level grouping for ColonBench.
    module = DATASETS[args.dataset]
    train_loader, val_loader = module.initialize_segmentation_dataloaders(
        data_dir=root, batch_size=args.batch_size, image_size=args.image_size, num_workers=0, seed=42)

    video_ids = {}
    if args.dataset == "colonbench":
        with (root / "manifest.csv").open(newline="") as f:
            for row in csv.DictReader(f):
                video_ids[Path(row["image_path"]).resolve()] = row["video_id"]

    def records(loader):
        rows = []
        for image, mask in loader.dataset.samples:
            image, mask = Path(image).resolve(), Path(mask).resolve()
            row = {"image": image.relative_to(root).as_posix(), "mask": mask.relative_to(root).as_posix()}
            if args.dataset == "colonbench":
                row["video_id"] = video_ids[image]
            rows.append(row)
        return rows

    split = {"dataset": args.dataset, "split_seed": 42, "train": records(train_loader), "val": records(val_loader)}
    temp = split_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(split, indent=2))
    temp.replace(split_path)
    print(f"Created split: {split_path}", flush=True)
    return split


def make_loaders(args, split):
    dataset_cls = DATASETS[args.dataset].SegmentationDataset

    def samples(rows):
        return [(args.data_dir / row["image"], args.data_dir / row["mask"]) for row in rows]

    train_ds = dataset_cls(samples(split["train"]), image_size=args.image_size, train=True)
    val_ds = dataset_cls(samples(split["val"]), image_size=args.image_size, train=False)
    options = {"batch_size": args.batch_size, "num_workers": args.workers, "pin_memory": torch.cuda.is_available()}

    if args.dataset == "colonbench":
        video_ids = [row["video_id"] for row in split["train"]]
        counts = Counter(video_ids)
        weights = torch.tensor([1.0 / counts[vid] for vid in video_ids], dtype=torch.double)
        sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, sampler=sampler, **options)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **options)

    return train_loader, DataLoader(val_ds, shuffle=False, **options)


@torch.inference_mode()
def evaluate_best(model, val_loader, val_records, edl, device):
    model.eval()
    rows = []
    for images, masks in tqdm(val_loader, desc="Evaluating best model", leave=False):
        images, masks = images.to(device), masks.to(device).long()
        logits = model(images).float()
        if edl:
            alpha = F.softplus(logits) + 1.0
            probs = alpha / alpha.sum(dim=1, keepdim=True)
        else:
            probs = logits.softmax(dim=1)

        dice, iou = compute_metrics(probs.argmax(dim=1), masks)
        entropy = torch.special.entr(probs).sum(dim=1)
        image_entropy = entropy.mean(dim=(1, 2))
        foreground = masks == 1
        fg_pixels = foreground.sum(dim=(1, 2))
        fg_entropy = (entropy * foreground).sum(dim=(1, 2)) / fg_pixels.clamp_min(1)
        fg_entropy = fg_entropy.masked_fill(fg_pixels == 0, float("nan"))

        batch = torch.stack([dice, iou, image_entropy, fg_entropy], dim=1).cpu().tolist()
        for d, j, h, fg_h in batch:
            record = val_records[len(rows)]
            row = {"image": record["image"], "mask": record["mask"], "dice": d, "iou": j,
                   "entropy": h, "foreground_entropy": fg_h}
            if "video_id" in record:
                row["video_id"] = record["video_id"]
            rows.append(row)

    if len(rows) != len(val_records):
        raise RuntimeError("Validation prediction count does not match the saved split.")
    return pd.DataFrame(rows)


def run_one(args, split, seed):
    set_seed(seed)
    edl = args.method == "edl"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = make_loaders(args, split)
    model = initialize_vit_unet(num_classes=2, ckpt_path=args.ckpt)

    with TemporaryDirectory(prefix="seg_multiseed_", dir="/tmp") as temp:
        history = train_segmentation(model, train_loader, val_loader, args.epochs, edl=edl, save_dir=temp)
        epochs = pd.DataFrame(history)
        # The trainer saves on strict improvement, so the first maximum is the saved epoch.
        best = epochs.loc[epochs["val_dice"].idxmax()]
        state = torch.load(Path(temp) / "best_model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        del state
        predictions = evaluate_best(model.to(device), val_loader, split["val"], edl, device)

    result = pd.DataFrame([{"best_epoch": int(best["epoch"]), "val_dice": predictions["dice"].mean(),
                            "val_iou": predictions["iou"].mean(), "mean_entropy": predictions["entropy"].mean(),
                            "mean_foreground_entropy": predictions["foreground_entropy"].mean()}])
    for frame in (predictions, epochs, result):
        frame.insert(0, "seed", seed)
        frame.insert(0, "method", args.method)
        frame.insert(0, "dataset", args.dataset)
    return predictions, epochs, result


def read_table(path):
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def replace_seed(previous, current, seed):
    if previous.empty:
        return current
    return pd.concat([previous.loc[previous["seed"] != seed], current], ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--method", choices=["ce", "edl"], required=True)
    parser.add_argument("--runs", type=int, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", "--bs", type=int, default=256)
    parser.add_argument("--image_size", type=int, default=224, choices=[224])
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    if not 1 <= args.runs <= 2**31 or args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("Invalid runs, epochs, batch size, or worker count.")

    args.data_dir = args.data_dir.expanduser().resolve()
    split = load_or_create_split(args)
    seeds = sample_seeds(args.runs)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    prediction_frames, epoch_frames, result_frames = [], [], []
    print(f"Dataset: {args.dataset} | Method: {args.method} | Seeds: {seeds}", flush=True)

    for seed in seeds:
        print(f"\nTraining seed {seed}", flush=True)
        predictions, epochs, result = run_one(args, split, seed)

        result_frames.append(result)
        results = pd.concat(result_frames, ignore_index=True)
        write_csv(results, out / "results_by_seed.csv")

        prediction_frames.append(predictions)
        write_csv(pd.concat(prediction_frames, ignore_index=True), out / "predictions.csv")

        epoch_frames.append(epochs)
        write_csv(pd.concat(epoch_frames, ignore_index=True), out / "epoch_metrics.csv")

        row = result.iloc[0]
        print(f"Seed {seed} | Best epoch: {int(row['best_epoch'])} | "
              f"Val Dice: {row['val_dice']:.4f} | Val IoU: {row['val_iou']:.4f}", flush=True)

    summary = results[METRICS].agg(["mean", "std"]).T.rename_axis("metric").reset_index()
    summary.insert(0, "runs", len(results))
    write_csv(summary, out / "results_mean_sd.csv")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()