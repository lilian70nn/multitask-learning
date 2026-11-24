import argparse
import hashlib
import json
import random
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification

from classification_vit.classification_dataloader import FindingDataset
from classification_vit.train_classification import initialize_vit_classifier
from classification_vit.hierarchical import CoarseDataset, SpecialistDataset, train_independent_model
from classification_vit.hierarchical_utils import SWIN_NAME, SwinInferenceDataset, build_coarse_mapping, calculate_metrics


METRICS = ["accuracy", "macro_precision", "macro_sensitivity", "macro_specificity", "macro_f1", "weighted_f1"]


def sample_seeds(count):
    # Same sequence across CE and EDL, including when --runs increases later.
    rng = random.Random(42)
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


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(args):
    split = json.loads(args.split_file.read_text())
    info = json.loads(args.groups_file.read_text())
    label_to_idx = split["label_to_idx"]
    names = sorted(label_to_idx, key=label_to_idx.get)

    if [label_to_idx[name] for name in names] != list(range(len(names))):
        raise ValueError("Expected consecutive class indices starting at zero.")
    if info["original_classes"] != {str(i): name for i, name in enumerate(names)}:
        raise ValueError("groups.json and dataset_split.json use different class mappings.")

    group_rows = sorted(info["merge_groups"], key=lambda row: row["group_id"])
    if [row["group_id"] for row in group_rows] != list(range(len(group_rows))):
        raise ValueError("Unexpected group IDs in groups.json.")

    groups = [row["class_indices"] for row in group_rows]
    members = [idx for group in groups for idx in group]
    if not groups or any(len(group) < 2 for group in groups):
        raise ValueError("Expected at least one merge group with two or more classes.")
    if len(set(members)) != len(members) or any(idx < 0 or idx >= len(names) for idx in members):
        raise ValueError("Invalid or overlapping classes in groups.json.")

    mapping = build_coarse_mapping(len(names), groups)
    for key, actual in zip(("leaf_to_coarse", "coarse_to_leaf", "group_to_coarse"), mapping[:3]):
        if actual != {int(k): int(v) for k, v in info[key].items()}:
            raise ValueError(f"Class mapping disagrees with groups.json: {key}")

    train = [(args.data_dir / path, int(label)) for path, label in split["train"]]
    val = [(args.data_dir / path, int(label)) for path, label in split["val"]]
    if not train or not val:
        raise ValueError("The saved split must contain both training and validation images.")
    return names, groups, mapping, train, val


def make_datasets(train, val):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    val_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    return FindingDataset(train, train_tf), FindingDataset(val, val_tf)


def train_and_record(model, train_loader, val_loader, epochs, run_dir, name, edl, swin=False):
    with TemporaryDirectory(prefix="cls_weights_", dir="/tmp") as temp:
        checkpoint = Path(temp) / "best.pt"
        model, _ = train_independent_model(model, train_loader, val_loader, epochs, checkpoint,
                                           edl=edl, swin=swin, name=name)
    return model.cpu()


def class_probabilities(logits, edl):
    logits = logits.float()
    if edl:
        alpha = F.softplus(logits) + 1.0
        return alpha / alpha.sum(dim=1, keepdim=True)
    return logits.softmax(dim=1)


@torch.inference_mode()
def evaluate_and_record(coarse, specialists, vit_loader, swin_loader, val, names, groups, mapping, edl, run_dir, device):
    _, coarse_to_leaf, group_to_coarse, _ = mapping
    coarse.to(device).eval()
    for specialist in specialists:
        specialist.to(device).eval()

    rows = []
    all_targets, all_preds = [], []
    for (vit_images, targets), (swin_images, swin_targets) in zip(vit_loader, swin_loader, strict=True):
        if not torch.equal(targets, swin_targets):
            raise ValueError("The two validation loaders are not aligned.")

        coarse_p = class_probabilities(coarse(vit_images.to(device)), edl)
        coarse_pred = coarse_p.argmax(dim=1)
        leaf_p = coarse_p.new_zeros((len(targets), len(names)))
        hard_pred = torch.full_like(coarse_pred, -1)

        # Ungrouped classes inherit their coarse probability.
        for coarse_idx, leaf_idx in coarse_to_leaf.items():
            leaf_p[:, leaf_idx] = coarse_p[:, coarse_idx]
            hard_pred[coarse_pred == coarse_idx] = leaf_idx

        # Full probabilities use all specialists. Hard predictions use only the routed branch.
        swin_images = swin_images.to(device)
        for group_id, group in enumerate(groups):
            coarse_idx = group_to_coarse[group_id]
            logits = specialists[group_id](pixel_values=swin_images).logits
            local_p = class_probabilities(logits, edl)
            leaf_p[:, group] = coarse_p[:, [coarse_idx]] * local_p

            routed = coarse_pred == coarse_idx
            leaf_indices = torch.as_tensor(group, device=device)
            hard_pred[routed] = leaf_indices[local_p[routed].argmax(dim=1)]

        if (hard_pred < 0).any():
            raise RuntimeError("Some images did not receive a hard-routed prediction.")
        if not torch.allclose(leaf_p.sum(dim=1), torch.ones_like(leaf_p[:, 0]), atol=1e-5):
            raise RuntimeError("The leaf probabilities do not sum to one.")

        entropy = torch.special.entr(leaf_p).sum(dim=1).cpu().tolist()
        target_list, pred_list = targets.tolist(), hard_pred.cpu().tolist()
        all_targets.extend(target_list)
        all_preds.extend(pred_list)

        for target, pred, h, probs in zip(target_list, pred_list, entropy, leaf_p.cpu().tolist()):
            path, expected_target = val[len(rows)]
            if target != expected_target:
                raise ValueError(f"Validation order mismatch at {path}")
            row = {"image_path": str(path), "true_label": names[target], "pred_label": names[pred],
                   "correct": int(target == pred), "entropy": h}
            row.update({f"prob_{name}": p for name, p in zip(names, probs)})
            rows.append(row)

    if len(rows) != len(val):
        raise ValueError("Validation prediction count does not match the saved split.")

    predictions = pd.DataFrame(rows)
    metrics, _ = calculate_metrics(all_targets, all_preds, len(names), class_names=names)
    scores = {key: float(metrics[key]) for key in METRICS}
    scores["mean_leaf_entropy"] = float(predictions["entropy"].mean())
    return predictions, scores


def run_one(args, seed, inputs):
    names, groups, mapping, train, val = inputs
    leaf_to_coarse, _, _, num_coarse = mapping
    edl = args.method == "edl"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = None  # 保留原有函数调用参数；组件训练文件使用 /tmp

    print(f"\nTraining {args.method.upper()} | seed={seed}", flush=True)
    set_seed(seed)
    options = {"num_workers": args.workers, "pin_memory": device.type == "cuda"}
    train_ds, val_ds = make_datasets(train, val)

    coarse_train = DataLoader(CoarseDataset(train_ds, leaf_to_coarse), batch_size=args.batch_size, shuffle=True, **options)
    coarse_val = DataLoader(CoarseDataset(val_ds, leaf_to_coarse), batch_size=args.batch_size, shuffle=False, **options)
    coarse = initialize_vit_classifier(num_coarse, ckpt_path=args.ckpt)
    coarse = train_and_record(coarse, coarse_train, coarse_val, args.epochs, run_dir, "Coarse", edl)

    processor = AutoImageProcessor.from_pretrained(SWIN_NAME)
    specialists = []
    for group_id, group in enumerate(groups):
        train_group = SpecialistDataset(train, processor, group)
        val_group = SpecialistDataset(val, processor, group)
        if not len(train_group) or not len(val_group):
            raise ValueError(f"Group {group_id} has no training or validation samples.")

        group_train = DataLoader(train_group, batch_size=16, shuffle=True, **options)
        group_val = DataLoader(val_group, batch_size=32, shuffle=False, **options)
        specialist = AutoModelForImageClassification.from_pretrained(SWIN_NAME, num_labels=len(group), ignore_mismatched_sizes=True)
        specialist = train_and_record(specialist, group_train, group_val, args.epochs, run_dir, f"Swin Group {group_id}", edl, swin=True)
        specialists.append(specialist)

    vit_loader = DataLoader(val_ds, batch_size=32, shuffle=False, **options)
    swin_loader = DataLoader(SwinInferenceDataset(val, processor), batch_size=32, shuffle=False, **options)
    return evaluate_and_record(coarse, specialists, vit_loader, swin_loader, val, names, groups, mapping, edl, run_dir, device)

def write_csv(frame, path):
    temp = path.with_suffix(".csv.tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def summarize(predictions, names, out):
    label_to_idx = {name: i for i, name in enumerate(names)}
    rows = []

    for (method, seed), group in predictions.groupby(["method", "seed"], sort=False):
        targets = group["true_label"].map(label_to_idx).to_numpy(dtype=int)
        preds = group["pred_label"].map(label_to_idx).to_numpy(dtype=int)
        metrics, _ = calculate_metrics(targets, preds, len(names), class_names=names)

        row = {"method": method, "seed": seed, **{key: float(metrics[key]) for key in METRICS}}
        row["mean_leaf_entropy"] = group["entropy"].mean()
        rows.append(row)

    runs = pd.DataFrame(rows)
    write_csv(runs, out / "results_by_seed.csv")

    summary = runs[METRICS + ["mean_leaf_entropy"]].agg(["mean", "std"]).T.rename_axis("metric").reset_index()
    write_csv(summary, out / "results_mean_sd.csv")
    print(summary.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["ce", "edl"], required=True)
    parser.add_argument("--runs", type=int, required=True)
    parser.add_argument("--groups_file", type=Path, required=True)
    parser.add_argument("--split_file", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    if not 1 <= args.runs <= 2**31 or args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("Invalid runs, epochs, batch size, or worker count.")

    inputs = load_inputs(args)
    seeds = sample_seeds(args.runs)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    prediction_frames = []
    metric_rows = []
    print(f"Method: {args.method.upper()} | Seeds: {seeds}", flush=True)

    for seed in seeds:
        predictions, scores = run_one(args, seed, inputs)

        metric_rows.append({"method": args.method, "seed": seed, **scores})
        results = pd.DataFrame(metric_rows)
        write_csv(results, out / "results_by_seed.csv")

        predictions.insert(0, "seed", seed)
        predictions.insert(0, "method", args.method)
        prediction_frames.append(predictions)
        write_csv(pd.concat(prediction_frames, ignore_index=True), out / "predictions.csv")

        summary = results[METRICS + ["mean_leaf_entropy"]].agg(["mean", "std"]).T
        write_csv(summary.rename_axis("metric").reset_index(), out / "results_mean_sd.csv")

        print(f"Seed {seed} | Val accuracy: {scores['accuracy']:.4f} | Val Macro-F1: {scores['macro_f1']:.4f}", flush=True)

    print(f"Saved to {out}")


if __name__ == "__main__":
    main()