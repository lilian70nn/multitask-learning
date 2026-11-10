
import argparse
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sacrebleu.metrics import BLEU
from bert_score import score as bert_score
from transformers import AutoModelForCausalLM

from caption.captioner_vit import load_captioning_model
from caption.train_captioner import CaptionDataset, split_dataset


CAPTION_TO_CLS = {
    "dyed-lifted-polyps": ["dyed-lifted-polyps"],
    "dyed-resection-margins": ["dyed-resection-margins"],
    "esophagitis": ["esophagitis-a", "esophagitis-b-d"],
    "normal-cecum": ["cecum"],
    "normal-pylorus": ["pylorus"],
    "normal-z-line": ["z-line"],
    "polyps": ["polyps"],
    "ulcerative-colitis": [
        "ulcerative-colitis-grade-1",
        "ulcerative-colitis-grade-2",
        "ulcerative-colitis-grade-3",
    ],
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def get_class_to_idx(cls_data_dir):
    paths = [
        p for p in cls_data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    counts = Counter(p.parent.name for p in paths)
    classes = sorted(name for name, count in counts.items() if count > 100)

    if len(classes) != 16:
        raise ValueError(f"Expected 16 classification classes, got {len(classes)}: {classes}")

    class_to_idx = {name: idx for idx, name in enumerate(classes)}

    for category, fine_classes in CAPTION_TO_CLS.items():
        for fine_class in fine_classes:
            if fine_class not in class_to_idx:
                raise ValueError(f"Missing classification class {fine_class!r} required by caption category {category!r}")

    print("Classification class order:")
    for name, idx in class_to_idx.items():
        print(f"  {idx}: {name}")

    return class_to_idx


def get_category(image_path):
    category = image_path.parent.name
    if category not in CAPTION_TO_CLS:
        raise ValueError(f"Unknown category: {category}, image: {image_path}")
    return category


def build_cls_gt(category, class_to_idx):
    fine_classes = CAPTION_TO_CLS[category]
    gt = torch.zeros(len(class_to_idx), dtype=torch.float32)
    weight = 1.0 / len(fine_classes)
    for name in fine_classes:
        gt[class_to_idx[name]] = weight
    return gt


def load_trained_model(args, device):
    model_args = SimpleNamespace(
        baseline=args.baseline,
        vit_ckpt=args.vit_ckpt,
        cls_ckpt=args.cls_ckpt,
        seg_ckpt=args.seg_ckpt,
        adapter_ckpt=None,
        qwen_name=args.qwen_name,
        seg_grid_size=args.seg_grid_size,
    )

    model = load_captioning_model(model_args).to(device)
    captioner_dir = Path(args.captioner_dir)
    adapter_state = torch.load(
        captioner_dir / "adapter.pth",
        map_location="cpu",
        weights_only=True,
    )
    model.adapter.load_state_dict(adapter_state, strict=True)

    del model.decoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model.decoder = AutoModelForCausalLM.from_pretrained(captioner_dir / "decoder").to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate(
    model,
    loader,
    annotations,
    class_to_idx,
    device,
    output_dir,
    max_new_tokens,
):
    predictions = []
    references = []
    image_ids = []
    categories = []

    for images, captions, ids, batch_categories in tqdm(
        loader, desc="Evaluating classification oracle"
    ):
        images = images.to(device, non_blocking=True)

        cls_gt = torch.stack([build_cls_gt(category, class_to_idx) for category in batch_categories]).to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda",):
            generated = model.generate(images, max_new_tokens=max_new_tokens, cls_gt=cls_gt)

        predictions.extend(text.strip() for text in generated)
        references.extend(captions)
        image_ids.extend(ids)
        categories.extend(batch_categories)

    bleu = BLEU(effective_order=True).corpus_score(predictions, [references]).score

    _, _, bert_f1 = bert_score(predictions, references, lang="en", verbose=False)

    metrics = {
        "BLEU": bleu,
        "BERTScore_F1": bert_f1.mean().item(),
        "val_samples": len(predictions),
    }

    pd.DataFrame({
        "image_id": image_ids,
        "category": categories,
        "reference": references,
        "prediction": predictions,
    }).to_csv(output_dir / "upper_bound_predictions.csv", index=False)

    with open(output_dir / "upper_bound_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics


class OracleDataset(torch.utils.data.Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        image, caption, image_id = self.subset[index]

        original_index = self.subset.indices[index]
        image_path = self.subset.dataset.samples[original_index][0]
        category = get_category(image_path)

        return image, caption, image_id, category


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--baseline", type=int, choices=[2, 3], required=True)
    parser.add_argument("--vit_ckpt", required=True)
    parser.add_argument("--cls_ckpt", required=True)
    parser.add_argument("--seg_ckpt", required=True)
    parser.add_argument("--captioner_dir", required=True)
    parser.add_argument("--data_dir", default=str(Path.home() / "mtl_data" / "kvasir-dataset-v2"),)
    parser.add_argument("--cls_data_dir", default=str(Path.home() / "mtl_data" / "labeled-images"))
    parser.add_argument("--qwen_name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--seg_grid_size", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    class_to_idx = get_class_to_idx(Path(args.cls_data_dir))
    with open(data_dir / "all_annotations.json", encoding="utf-8",) as f:
        annotations = json.load(f)

    dataset = CaptionDataset(data_dir)
    _, val_set = split_dataset(dataset)

    oracle_val_set = OracleDataset(val_set)

    val_loader = DataLoader(
        oracle_val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    model = load_trained_model(args, device)

    num_classes = model.adapter.cls.proj.in_features
    if num_classes != len(class_to_idx):
        raise ValueError(f"Classifier checkpoint has {num_classes} classes, but GT mapping has {len(class_to_idx)}.")

    metrics = evaluate(
        model,
        val_loader,
        annotations,
        class_to_idx,
        device,
        output_dir,
        args.max_new_tokens,
    )

    print(f"Baseline: {args.baseline}")
    print(f"BLEU: {metrics['BLEU']:.4f}")
    print(f"BERTScore F1: {metrics['BERTScore_F1']:.4f}")
    print(f"Validation samples: {metrics['val_samples']}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
