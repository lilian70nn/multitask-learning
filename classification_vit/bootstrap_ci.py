import argparse
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from sklearn.metrics import precision_recall_fscore_support
from classification_vit.classification_dataloader import initialize_classification_dataloaders
from classification_vit.hierarchical import HierarchicalClassifier
from classification_vit.hierarchical_utils import calculate_metrics, save_csv


@torch.inference_mode()
def collect_predictions(model, loader, device):
    targets, preds = [], []
    for images, labels in tqdm(loader, desc="Hierarchical inference"):
        logits = model(images.to(device))
        predictions = logits.argmax(dim=1)
        targets.extend(labels.tolist())
        preds.extend(predictions.cpu().tolist())
    return targets, preds


def bootstrap_ci(targets, preds, class_names, n_bootstrap=1000, confidence=0.95, seed=42):
    targets, preds = np.asarray(targets), np.asarray(preds)
    num_classes, n = len(class_names), len(targets)
    labels = np.arange(num_classes)
    rng = np.random.default_rng(seed)
    original, _ = calculate_metrics(targets, preds, num_classes, class_names=class_names)
    report = original["classification_report"]

    per_class_scores = np.full((n_bootstrap, num_classes, 3), np.nan)
    overall_names = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
    overall_scores = np.zeros((n_bootstrap, len(overall_names)))

    for b in tqdm(range(n_bootstrap), desc="Bootstrap"):
        indices = rng.choice(n, size=n, replace=True)
        y_true, y_pred = targets[indices], preds[indices]

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )

        per_class_scores[b, :, 0] = precision
        per_class_scores[b, :, 1] = recall
        per_class_scores[b, :, 2] = f1
        per_class_scores[b, support == 0, :] = np.nan

        overall_scores[b] = [
            np.mean(y_true == y_pred),
            precision.mean(),
            recall.mean(),
            f1.mean(),
            np.average(f1, weights=support),
        ]

    alpha = (1 - confidence) / 2
    per_class_rows = []

    for i, name in enumerate(class_names):
        row = {"class": name, "support": int(report[name]["support"])}

        for j, metric in enumerate(["precision", "recall", "f1"]):
            values = per_class_scores[:, i, j]
            values = values[~np.isnan(values)]
            if len(values) == 0:
                raise ValueError(f"No valid bootstrap samples for class {name}")

            row[metric] = float(report[name][metric])
            row[f"{metric}_ci_lower"] = float(np.quantile(values, alpha))
            row[f"{metric}_ci_upper"] = float(np.quantile(values, 1 - alpha))

        row["valid_bootstrap_draws"] = int(np.sum(~np.isnan(per_class_scores[:, i, 0])))
        per_class_rows.append(row)

    overall_rows = []
    for j, metric in enumerate(overall_names):
        overall_rows.append({
            "metric": metric,
            "value": float(original[metric]),
            "ci_lower": float(np.quantile(overall_scores[:, j], alpha)),
            "ci_upper": float(np.quantile(overall_scores[:, j], 1 - alpha)),
        })

    return per_class_rows, overall_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/content/labeled-images")
    parser.add_argument("--ce_dir", type=str, default=None)
    parser.add_argument("--edl_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.ce_dir and not args.edl_dir:
        parser.error("Please provide --ce_dir and/or --edl_dir")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader, label_to_idx = initialize_classification_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size, seed=args.seed
    )

    class_names = [name for name, _ in sorted(label_to_idx.items(), key=lambda x: x[1])]
    num_classes = len(class_names)

    for model_name, ckpt_dir, edl in [("CE", args.ce_dir, False), ("EDL", args.edl_dir, True)]:
        if ckpt_dir is None:
            continue

        save_dir = Path(ckpt_dir).parent / "bootstrap"
        save_dir.mkdir(parents=True, exist_ok=True)

        model = HierarchicalClassifier(ckpt_dir=ckpt_dir, edl=edl).to(device).eval()
        if model.num_classes != num_classes:
            raise ValueError(f"{model_name}: model has {model.num_classes} classes, dataset has {num_classes}")

        targets, preds = collect_predictions(model, val_loader, device)
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

        per_class, overall = bootstrap_ci(
            targets, preds, class_names, n_bootstrap=args.n_bootstrap, seed=args.seed
        )

        save_csv(save_dir / "per_class_ci.csv", per_class)
        save_csv(save_dir / "overall_ci.csv", overall)
        print(f"\nAll results saved to: {save_dir}")

        print(f"\n{model_name} Hierarchical Validation Results (95% Bootstrap CI):")
        for row in overall:
            print(f"{row['metric']}: {row['value']:.4f} [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]")

        print("\nPer-class F1:")
        for row in per_class:
            print(f"{row['class']}: {row['f1']:.4f} [{row['f1_ci_lower']:.4f}, {row['f1_ci_upper']:.4f}]")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    
if __name__ == "__main__":
    main()
