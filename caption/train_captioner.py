import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sacrebleu.metrics import BLEU
from bert_score import score as bert_score
from transformers import AutoModelForCausalLM

from caption.captioner_vit import load_captioning_model


# ==================== CONFIG ====================
DATA_DIR = Path("/content/segmented-images")
CSV_FILE = DATA_DIR / "pseudo_captions.csv"
IMAGE_DIR = DATA_DIR / "images"

IMAGE_SIZE = 224
TRAIN_RATIO = 0.8
SEED = 42


class CaptionDataset(Dataset):
    def __init__(self, csv_file, image_dir):
        self.df = pd.read_csv(csv_file).dropna(subset=["image_id", "caption"]).reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.samples = []
        for _, row in self.df.iterrows():
            image_id = str(row["image_id"])
            image_path = self.find_image(image_id)
            if image_path is not None:
                self.samples.append((image_path, str(row["caption"]), image_id))

        if not self.samples:
            raise ValueError(f"No valid images found in {self.image_dir}")

    def find_image(self, image_id):
        for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]:
            path = self.image_dir / f"{image_id}{ext}"
            if path.exists():
                return path
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, caption, image_id = self.samples[idx]
        with Image.open(image_path) as img:
            image = self.transform(img.convert("RGB"))
        return image, caption, image_id


def split_dataset(dataset):
    indices = list(range(len(dataset)))
    random.Random(SEED).shuffle(indices)

    n_train = int(len(indices) * TRAIN_RATIO)

    train_set = Subset(dataset, indices[:n_train])
    val_set = Subset(dataset, indices[n_train:])

    if not train_set or not val_set:
        raise ValueError("Dataset is too small for train/validation splitting.")

    return train_set, val_set


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    use_amp = device.type == "cuda"

    pbar = tqdm(loader, desc="Training", leave=False)

    for images, captions, _ in pbar:
        images = images.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            loss = model(images, captions).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    use_amp = device.type == "cuda"

    for images, captions, _ in tqdm(loader, desc="Validation", leave=False):
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            loss = model(images, captions).loss
        total_loss += loss.item()

    return total_loss / len(loader)


def save_model(model, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.adapter.state_dict(), output_dir / "adapter.pth")
    model.decoder.save_pretrained(output_dir / "decoder", safe_serialization=True)
    model.tokenizer.save_pretrained(output_dir / "decoder")


def load_best_model(model, output_dir, device):
    adapter_state = torch.load(output_dir / "adapter.pth", map_location="cpu", weights_only=True)
    model.adapter.load_state_dict(adapter_state)

    del model.decoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model.decoder = AutoModelForCausalLM.from_pretrained(output_dir / "decoder").to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, loader, device, output_dir, max_new_tokens):
    model.eval()
    predictions, references, image_ids = [], [], []

    for images, captions, ids in tqdm(loader, desc="Evaluating"):
        images = images.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            generated = model.generate(images, max_new_tokens=max_new_tokens)

        predictions.extend([text.strip() for text in generated])
        references.extend(list(captions))
        image_ids.extend(list(ids))

    bleu = BLEU(effective_order=True).corpus_score(predictions, [references]).score
    _, _, bert_f1 = bert_score(predictions, references, lang="en", verbose=False)
    bertscore = bert_f1.mean().item()

    pd.DataFrame({
        "image_id": image_ids,
        "reference": references,
        "prediction": predictions
    }).to_csv(output_dir / "val_predictions.csv", index=False)

    metrics = {"BLEU": bleu, "BERTScore_F1": bertscore, "val_samples": len(predictions)}

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--baseline", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--vit_ckpt", type=str, required=True)
    parser.add_argument("--cls_ckpt", type=str, default=None)
    parser.add_argument("--seg_ckpt", type=str, default=None)
    parser.add_argument("--qwen_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--seg_grid_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="captioner_checkpoints")

    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir) / f"baseline_{args.baseline}"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = CaptionDataset(CSV_FILE, IMAGE_DIR)
    train_set, val_set = split_dataset(dataset)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Baseline {args.baseline} | Train: {len(train_set)} | Val: {len(val_set)}")

    model_args = SimpleNamespace(
        baseline=args.baseline,
        vit_ckpt=args.vit_ckpt,
        cls_ckpt=args.cls_ckpt,
        seg_ckpt=args.seg_ckpt,
        adapter_ckpt=None,
        qwen_name=args.qwen_name,
        seg_grid_size=args.seg_grid_size
    )

    model = load_captioning_model(model_args).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, output_dir)
            print("Best model saved.")

    del optimizer
    model = load_best_model(model, output_dir, device)

    metrics = evaluate(model, val_loader, device, output_dir, args.max_new_tokens)

    config = {
        "baseline": args.baseline,
        "vit_ckpt": args.vit_ckpt,
        "cls_ckpt": args.cls_ckpt,
        "seg_ckpt": args.seg_ckpt,
        "qwen_name": args.qwen_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seed": SEED,
        "best_val_loss": best_val_loss
    }

    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=4)

    print(f"\nFinished Baseline {args.baseline}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"BLEU: {metrics['BLEU']:.4f}")
    print(f"BERTScore F1: {metrics['BERTScore_F1']:.4f}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()