import argparse
from pathlib import Path
import numpy as np
import json
import torch
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from classification_vit.classification_dataloader import FindingDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from classification_vit.hierarchical import HierarchicalClassifier
from classification_vit.hierarchical_utils import calculate_metrics, save_csv
from classification_vit.hierarchical_utils import evaluate_hierarchical


@torch.inference_mode()
def collect_predictions(model, loader, device):
    targets, preds = [], []
    for images, labels in tqdm(loader, desc="Hierarchical inference"):
        logits = model(images.to(device))
        predictions = logits.argmax(dim=1)
        targets.extend(labels.tolist())
        preds.extend(predictions.cpu().tolist())
    return targets, preds


def specificity_from_cm(cm):
    negatives = cm.sum() - cm.sum(axis=1)
    false_positives = cm.sum(axis=0) - np.diag(cm)
    true_negatives = negatives - false_positives

    return np.divide(
        true_negatives,
        negatives,
        out=np.full(len(cm), np.nan),
        where=negatives > 0,
    )


def bootstrap_ci(
    targets, preds, class_names, n_bootstrap=1000, confidence=0.95, seed=42
):
    targets, preds = np.asarray(targets), np.asarray(preds)
    num_classes, n = len(class_names), len(targets)
    labels = np.arange(num_classes)
    rng = np.random.default_rng(seed)
    alpha = (1 - confidence) / 2

    original, cm = calculate_metrics(
        targets, preds, num_classes, class_names=class_names
    )
    report = original["classification_report"]
    specificity = specificity_from_cm(cm)
    original["macro_sensitivity"] = original["macro_recall"]
    original["macro_specificity"] = float(np.nanmean(specificity))

    per_class_names = ["precision", "sensitivity", "f1-score", "specificity"]
    overall_names = [
        "accuracy", "macro_precision", "macro_sensitivity",
        "macro_f1", "weighted_f1", "macro_specificity",
    ]
    per_class_scores = np.full((n_bootstrap, num_classes, 4), np.nan)
    overall_scores = np.full((n_bootstrap, len(overall_names)), np.nan)

    for b in tqdm(range(n_bootstrap), desc="Bootstrap"):
        indices = rng.choice(n, size=n, replace=True)
        y_true, y_pred = targets[indices], preds[indices]

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )
        draw_specificity = specificity_from_cm(
            confusion_matrix(y_true, y_pred, labels=labels)
        )

        per_class_scores[b, :, :3] = np.column_stack((precision, recall, f1))
        per_class_scores[b, support == 0, :3] = np.nan
        per_class_scores[b, :, 3] = draw_specificity

        overall_scores[b] = [
            np.mean(y_true == y_pred),
            precision.mean(),
            recall.mean(),
            f1.mean(),
            np.average(f1, weights=support),
            np.nanmean(draw_specificity),
        ]

    def interval(values):
        values = values[np.isfinite(values)]
        if values.size == 0:
            return np.nan, np.nan
        lower, upper = np.quantile(values, [alpha, 1 - alpha])
        return float(lower), float(upper)

    per_class_rows = []
    for i, name in enumerate(class_names):
        stats = report[name]
        point_estimates = [
            stats["precision"], stats["recall"], stats["f1-score"], specificity[i]
        ]
        row = {"class": name, "support": int(stats["support"])}
        for j, metric in enumerate(per_class_names):
            lower, upper = interval(per_class_scores[:, i, j])
            row[metric] = float(point_estimates[j])
            row[f"{metric}_ci_lower"] = lower
            row[f"{metric}_ci_upper"] = upper

        row["valid_bootstrap_draws"] = int(
            np.isfinite(per_class_scores[:, i, 0]).sum()
        )
        per_class_rows.append(row)

    overall_rows = []
    for j, metric in enumerate(overall_names):
        lower, upper = interval(overall_scores[:, j])
        overall_rows.append({
            "metric": metric,
            "value": float(original[metric]),
            "ci_lower": lower,
            "ci_upper": upper,
        })

    return per_class_rows, overall_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/content/labeled-images")
    parser.add_argument("--split_file", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--edl", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.split_file, encoding="utf-8") as f:
        split_info = json.load(f)

    label_to_idx = split_info["label_to_idx"]

    val_samples = [
        (Path(args.data_dir) / relative_path, int(label))
        for relative_path, label in split_info["val"]
    ]

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_loader = DataLoader(
        FindingDataset(val_samples, val_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    class_names = [name for name, _ in sorted(label_to_idx.items(), key=lambda x: x[1])]
    num_classes = len(class_names)

    ckpt_dir = Path(args.ckpt_dir)

    with open(ckpt_dir / "groups.json") as f:
        groups_info = json.load(f)

    model_class_names = [groups_info["original_classes"][str(i)] for i in range(num_classes)]
    if model_class_names != class_names:
        raise ValueError("Class mapping does not match the trained model.")

    save_dir = ckpt_dir.parent / "bootstrap"
    save_dir.mkdir(parents=True, exist_ok=True)

    model = HierarchicalClassifier(ckpt_dir=ckpt_dir, edl=args.edl).to(device).eval()

    # 按相同顺序组织专家及其类别组
    group_ids = sorted(model.merge_groups)

    specialists = [model.specialists[str(group_id)] for group_id in group_ids]
    merge_groups = [model.merge_groups[group_id] for group_id in group_ids]
    group_to_coarse = {index: model.group_to_coarse[group_id] for index, group_id in enumerate(group_ids)}

    (
        final_metrics,
        final_cm,
        targets,
        preds,
        coarse_preds,
        coarse_metrics,
        coarse_cm,
    ) = evaluate_hierarchical(
        coarse_model=model.coarse,
        specialists=specialists,
        val_loader=val_loader,
        val_samples=val_samples,
        processor=model.processor,
        coarse_to_leaf=model.coarse_to_leaf,
        group_to_coarse=group_to_coarse,
        merge_groups=merge_groups,
        num_classes=num_classes,
        edl=args.edl,
        batch_size=args.batch_size,
    )

    print(
        f"Final Hierarchical | "
        f"Acc: {final_metrics['accuracy']:.4f} | "
        f"Macro-F1: {final_metrics['macro_f1']:.4f}"
    )

    prediction_rows = [
        {
            "image_path": str(val_loader.dataset.samples[i][0]),
            "true_label": class_names[int(targets[i])],
            "pred_label": class_names[int(preds[i])],
            "correct": int(targets[i] == preds[i]),
        }
        for i in range(len(targets))
    ]
    save_csv(save_dir / "predictions.csv", prediction_rows)

    per_class, overall = bootstrap_ci(targets, preds, class_names, n_bootstrap=args.n_bootstrap, seed=args.seed)
    save_csv(save_dir / "per_class_ci.csv", per_class)
    save_csv(save_dir / "overall_ci.csv", overall)

    print(f"Results saved to: {save_dir}")

    
if __name__ == "__main__":
    main()
