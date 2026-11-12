"""Independent ViT coarse classifier and Swin specialists with hard-routed inference."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification

from classification_vit.train_classification import compute_loss, get_predictions, initialize_vit_classifier
from classification_vit.hierarchical_utils import (
    SWIN_NAME, SwinInferenceDataset, build_coarse_mapping, calculate_metrics,
    discover_groups, evaluate_hierarchical, evaluate_model, evaluate_specialists,
    save_csv,
)


class CoarseDataset(Dataset):
    def __init__(self, dataset, leaf_to_coarse):
        self.dataset = dataset
        self.leaf_to_coarse = leaf_to_coarse

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, self.leaf_to_coarse[int(label)]


class SpecialistDataset(Dataset):
    def __init__(self, samples, processor, group):
        self.dataset = SwinInferenceDataset([(path, group.index(label)) for path, label in samples if label in group], processor)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


def train_independent_model(model, train_loader, val_loader, epochs, checkpoint_path, edl=False, swin=False, name="Model"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best_f1, history = -1.0, []
    num_classes = model.config.num_labels if swin else model.head.out_features

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_count = 0.0, 0
        for images, targets in tqdm(train_loader, desc=f"{name} Epoch {epoch}/{epochs}"):
            images, targets = images.to(device), targets.to(device).long()
            optimizer.zero_grad()
            logits = model(pixel_values=images).logits if swin else model(images)
            loss = compute_loss(logits, targets, edl)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * targets.size(0)
            total_count += targets.size(0)

        model.eval()
        targets_all, preds_all, uncertainties = [], [], []
        val_loss, val_count = 0.0, 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device).long()
                logits = model(pixel_values=images).logits if swin else model(images)
                loss = compute_loss(logits, targets, edl)
                preds, uncertainty = get_predictions(logits, edl)
                val_loss += loss.item() * targets.size(0)
                val_count += targets.size(0)
                targets_all.extend(targets.cpu().tolist())
                preds_all.extend(preds.cpu().tolist())
                if uncertainty is not None:
                    uncertainties.extend(uncertainty.cpu().tolist())

        metrics, _ = calculate_metrics(targets_all, preds_all, num_classes, uncertainties if edl else None)
        row = {"epoch": epoch, "train_loss": total_loss / max(total_count, 1), "val_loss": val_loss / max(val_count, 1),
               "val_accuracy": metrics["accuracy"], "val_macro_precision": metrics["macro_precision"],
               "val_macro_recall": metrics["macro_recall"], "val_macro_f1": metrics["macro_f1"],
               "val_weighted_f1": metrics["weighted_f1"], "val_uncertainty": metrics["mean_uncertainty"]}
        history.append(row)
        save_csv(checkpoint_path.parent / f"{name.lower().replace(' ', '_')}_metrics.csv", history)
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            torch.save(model.state_dict(), checkpoint_path)
        print(f"{name} Epoch {epoch}/{epochs} | Train Loss: {row['train_loss']:.4f} | Val Acc: {metrics['accuracy']:.4f} | Val Macro-F1: {metrics['macro_f1']:.4f}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    print(f"{name} finished | Best Val Macro-F1: {best_f1:.4f}")
    return model, history


def train_hierarchical_classifier(model, train_loader, val_loader, label_to_idx, ckpt_path=None, single_stage_ckpt=None,
                                  edl=False, save_dir="./results/classification/hierarchical", epochs=20, threshold=15):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(label_to_idx)
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    if single_stage_ckpt is None:
        raise ValueError("single_stage_ckpt must be provided.")
    if isinstance(val_loader.sampler, torch.utils.data.RandomSampler):
        raise ValueError("val_loader must have shuffle=False to align predictions with val_samples.")

    model.load_state_dict(torch.load(single_stage_ckpt, map_location="cpu", weights_only=True))
    model = model.to(device)
    single_metrics, single_targets, single_preds = evaluate_model(model, val_loader, device, edl=edl)
    single_metrics_full, single_cm = calculate_metrics(single_targets, single_preds, num_classes)
    single_metrics_full["loss"] = single_metrics["loss"]
    single_metrics_full["mean_uncertainty"] = single_metrics["uncertainty"]
    np.savetxt(save_dir / "single_stage_confusion_matrix.csv", single_cm, fmt="%d", delimiter=",")
    print(f"Single-stage | Acc: {single_metrics_full['accuracy']:.4f} | Macro-F1: {single_metrics_full['macro_f1']:.4f}")

    merge_groups = discover_groups(single_cm, threshold=threshold)
    print(f"Detected merge groups: {[[idx_to_label[i] for i in group] for group in merge_groups]}")
    if not merge_groups:
        print("No merge groups detected. Hierarchical training skipped.")
        with open(save_dir / "final_metrics.json", "w") as f:
            json.dump({"single_stage": single_metrics_full, "hierarchical": None, "reason": "No merge groups detected"}, f, indent=2)
        return

    leaf_to_coarse, coarse_to_leaf, group_to_coarse, num_coarse = build_coarse_mapping(num_classes, merge_groups)
    groups_info = {"threshold": threshold, "original_classes": {str(i): idx_to_label[i] for i in range(num_classes)},
                   "merge_groups": [{"group_id": i, "class_indices": group, "class_names": [idx_to_label[j] for j in group]} for i, group in enumerate(merge_groups)],
                   "leaf_to_coarse": {str(k): v for k, v in leaf_to_coarse.items()},
                   "coarse_to_leaf": {str(k): v for k, v in coarse_to_leaf.items()},
                   "group_to_coarse": {str(k): v for k, v in group_to_coarse.items()}, "num_coarse_classes": num_coarse}
    with open(save_dir / "groups.json", "w") as f:
        json.dump(groups_info, f, indent=2)

    train_samples, val_samples = train_loader.dataset.samples, val_loader.dataset.samples
    if len(val_samples) != len(single_targets):
        raise ValueError("Validation dataset and predictions have different lengths.")
    model.to("cpu")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    coarse_model = initialize_vit_classifier(num_coarse, ckpt_path=ckpt_path)
    coarse_train_loader = DataLoader(CoarseDataset(train_loader.dataset, leaf_to_coarse), batch_size=train_loader.batch_size,
                                     shuffle=True, num_workers=train_loader.num_workers, pin_memory=True)
    coarse_val_loader = DataLoader(CoarseDataset(val_loader.dataset, leaf_to_coarse), batch_size=val_loader.batch_size,
                                   shuffle=False, num_workers=val_loader.num_workers, pin_memory=True)
    coarse_model, coarse_history = train_independent_model(coarse_model, coarse_train_loader, coarse_val_loader, epochs,
                                                           save_dir / "coarse_best_model.pt", edl=edl, name="Coarse")

    processor = AutoImageProcessor.from_pretrained(SWIN_NAME)
    specialists, specialist_histories = [], {}
    for group_id, group in enumerate(merge_groups):
        specialist = AutoModelForImageClassification.from_pretrained(SWIN_NAME, num_labels=len(group), ignore_mismatched_sizes=True)
        group_train_dataset = SpecialistDataset(train_samples, processor, group)
        group_val_dataset = SpecialistDataset(val_samples, processor, group)
        if not len(group_train_dataset) or not len(group_val_dataset):
            raise ValueError(f"Group {group_id} has no train or validation samples.")
        group_train_loader = DataLoader(group_train_dataset, batch_size=16, shuffle=True, num_workers=train_loader.num_workers, pin_memory=True)
        group_val_loader = DataLoader(group_val_dataset, batch_size=32, shuffle=False, num_workers=val_loader.num_workers, pin_memory=True)
        specialist, history = train_independent_model(specialist, group_train_loader, group_val_loader, epochs,
                                                      save_dir / f"swin_group_{group_id}_best_model.pt", edl=edl,
                                                      swin=True, name=f"Swin Group {group_id}")
        specialists.append(specialist.to("cpu"))
        if device.type == "cuda":
            torch.cuda.empty_cache()
        specialist_histories[f"group_{group_id}"] = history

    specialists = [specialist.to(device) for specialist in specialists]
    hierarchical_metrics, hierarchical_cm, targets, final_preds, coarse_preds, coarse_metrics, coarse_cm = evaluate_hierarchical(
        coarse_model, specialists, val_loader, val_samples, processor, coarse_to_leaf, group_to_coarse, merge_groups, num_classes, edl=edl)
    specialist_metrics, specialist_cms = evaluate_specialists(specialists, val_samples, processor, merge_groups, edl=edl)
    hierarchical_metrics.pop("mean_uncertainty", None)

    np.savetxt(save_dir / "coarse_confusion_matrix.csv", coarse_cm, fmt="%d", delimiter=",")
    np.savetxt(save_dir / "hierarchical_confusion_matrix.csv", hierarchical_cm, fmt="%d", delimiter=",")
    for group_name, cm in specialist_cms.items():
        np.savetxt(save_dir / f"swin_{group_name}_confusion_matrix.csv", cm, fmt="%d", delimiter=",")

    prediction_rows = [{"image_path": str(val_samples[i][0]), "true_label": idx_to_label[targets[i]],
                        "coarse_prediction": coarse_preds[i], "final_prediction": idx_to_label[final_preds[i]],
                        "correct": int(targets[i] == final_preds[i])} for i in range(len(targets))]
    save_csv(save_dir / "hierarchical_predictions.csv", prediction_rows)
    final_metrics = {"single_stage": single_metrics_full, "coarse": coarse_metrics, "specialists": specialist_metrics,
                     "hierarchical": hierarchical_metrics, "training_history": {"coarse": coarse_history, "specialists": specialist_histories}}
    with open(save_dir / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    print("\n========== Final Results ==========")
    print(f"Single-stage | Acc: {single_metrics_full['accuracy']:.4f} | Macro-F1: {single_metrics_full['macro_f1']:.4f}")
    print(f"Coarse | Acc: {coarse_metrics['accuracy']:.4f} | Macro-F1: {coarse_metrics['macro_f1']:.4f}")
    for group_name, metrics in specialist_metrics.items():
        print(f"Swin {group_name} | Acc: {metrics['accuracy']:.4f} | Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Final Hierarchical | Acc: {hierarchical_metrics['accuracy']:.4f} | Macro-F1: {hierarchical_metrics['macro_f1']:.4f}")
    print(f"Results saved to: {save_dir}")
    return final_metrics



class HierarchicalClassifier(nn.Module):
    def __init__(self, ckpt_dir, edl=False):
        super().__init__()
        ckpt_dir = Path(ckpt_dir)
        self.edl = edl

        with open(ckpt_dir / "groups.json") as f:
            info = json.load(f)

        self.num_classes = len(info["original_classes"])
        self.coarse_to_leaf = {int(k): int(v)for k, v in info["coarse_to_leaf"].items()}
        self.group_to_coarse = {int(k): int(v) for k, v in info["group_to_coarse"].items()}
        self.merge_groups = {int(group["group_id"]): group["class_indices"] for group in info["merge_groups"]}

        # Stage 1: load trained Coarse ViT
        self.coarse = initialize_vit_classifier(num_classes=info["num_coarse_classes"], ckpt_path=None)
        self.coarse.load_state_dict(
            torch.load(ckpt_dir / "coarse_best_model.pt", map_location="cpu", weights_only=True),
            strict=True,
        )

        # Stage 2: load trained Swin Specialists
        self.processor = AutoImageProcessor.from_pretrained(SWIN_NAME)
        self.specialists = nn.ModuleDict()

        for group_id, classes in self.merge_groups.items():
            specialist = AutoModelForImageClassification.from_pretrained(SWIN_NAME, num_labels=len(classes), ignore_mismatched_sizes=True)
            specialist.load_state_dict(
                torch.load(ckpt_dir / f"swin_group_{group_id}_best_model.pt", map_location="cpu", weights_only=True,),
                strict=True,
            )
            self.specialists[str(group_id)] = specialist

    def preprocess_swin(self, images):
        # Reverse Captioner's ImageNet normalization.
        mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        rgb = (images * std + mean).clamp(0, 1)

        size = self.processor.size
        if "shortest_edge" in size:
            shortest = size["shortest_edge"]
            h, w = rgb.shape[-2:]
            scale = shortest / min(h, w)
            new_h = round(h * scale)
            new_w = round(w * scale)
        else:
            new_h = size["height"]
            new_w = size["width"]

        rgb = F.interpolate(rgb, size=(new_h, new_w), mode="bicubic", align_corners=False).clamp(0, 1)
        if getattr(self.processor, "do_center_crop", False):
            crop_h = self.processor.crop_size["height"]
            crop_w = self.processor.crop_size["width"]
            top = (new_h - crop_h) // 2
            left = (new_w - crop_w) // 2
            rgb = rgb[:, :, top:top + crop_h, left:left + crop_w]

        swin_mean = images.new_tensor(self.processor.image_mean).view(1, 3, 1, 1)
        swin_std = images.new_tensor(self.processor.image_std).view(1, 3, 1, 1)

        return (rgb - swin_mean) / swin_std

    @torch.no_grad()
    def forward(self, images):
        # Stage 1
        coarse_logits = self.coarse(images)

        if self.edl:
            coarse_alpha = F.softplus(coarse_logits) + 1
            coarse_probs = coarse_alpha / coarse_alpha.sum(dim=1, keepdim=True)
        else:
            coarse_probs = F.softmax(coarse_logits, dim=1)

        coarse_preds = coarse_probs.argmax(dim=1)
        final_logits = torch.full((images.size(0), self.num_classes), -1e4, device=images.device, dtype=coarse_logits.dtype)
        for coarse_idx, leaf_idx in self.coarse_to_leaf.items():
            mask = coarse_preds == coarse_idx
            final_logits[mask, leaf_idx] = coarse_logits[mask, coarse_idx]

        swin_images = None
        for group_id, classes in self.merge_groups.items():
            coarse_idx = self.group_to_coarse[group_id]
            indices = torch.where(coarse_preds == coarse_idx)[0]

            if indices.numel() == 0:
                continue

            if swin_images is None:
                swin_images = self.preprocess_swin(images)

            logits = self.specialists[str(group_id)](pixel_values=swin_images[indices]).logits
            class_indices = torch.tensor(classes, dtype=torch.long, device=images.device)
            final_logits[indices[:, None], class_indices[None, :],] = logits

        return final_logits
