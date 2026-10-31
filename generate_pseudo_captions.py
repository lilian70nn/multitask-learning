import json
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

DATA_DIR = Path("/content/segmented-images")
IMAGE_DIR = DATA_DIR / "images"
MASK_DIR = DATA_DIR / "masks"
ANNOTATION_FILE = DATA_DIR / "bounding-boxes.json"
OUTPUT_FILE = Path("/content/drive/MyDrive/pseudo_captions.csv")
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"


def find_file(directory, image_id):
    for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]:
        path = directory / f"{image_id}{ext}"
        if path.exists():
            return path
    return None


def make_overlay(image, mask):
    image = image.convert("RGB")
    mask = mask.convert("L").resize(image.size, Image.Resampling.NEAREST)
    overlay = Image.new("RGB", image.size, (255, 0, 0))
    result = Image.blend(image, overlay, 0.4)
    return Image.composite(result, image, mask.point(lambda p: 255 if p > 127 else 0))


def generate_caption(image, overlay, labels, model, processor):
    label_text = ", ".join(labels)
    prompt = (
        f"You are describing a gastrointestinal endoscopy image. "
        f"The first image is the original endoscopy image. "
        f"The second image is the same image with the ground-truth lesion segmentation highlighted in red. "
        f"The annotated lesion label is: {label_text}. "
        f"Write one concise, clinically descriptive English caption describing the visible finding and its location or appearance. "
        f"Use the provided label and segmentation to identify the annotated finding. "
        f"Do not invent diagnoses, histology, severity, or other details that cannot be established from the images. "
        f"Output only the caption."
    )
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "image", "image": overlay},
        {"type": "text", "text": prompt}
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image, overlay], return_tensors="pt", padding=True).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)

    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    with open(ANNOTATION_FILE, "r") as f:
        annotations = json.load(f)

    rows = []
    if OUTPUT_FILE.exists():
        rows = pd.read_csv(OUTPUT_FILE).fillna("").to_dict("records")

    completed = {row["image_id"] for row in rows if row.get("caption")}

    for index, (image_id, info) in enumerate(annotations.items(), start=1):
        if image_id in completed:
            continue

        image_path = find_file(IMAGE_DIR, image_id)
        mask_path = find_file(MASK_DIR, image_id)
        labels = sorted({box["label"] for box in info.get("bbox", []) if box.get("label")})

        if image_path is None or mask_path is None or not labels:
            print(f"[{index}/{len(annotations)}] Skipped {image_id}: missing image, mask, or label")
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path)
            overlay = make_overlay(image, mask)
            caption = generate_caption(image, overlay, labels, model, processor)

            if not caption:
                print(f"[{index}/{len(annotations)}] Empty caption: {image_id}")
                continue

            rows.append({
                "image_id": image_id,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "label": ", ".join(labels),
                "caption": caption
            })

            pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False)
            print(f"[{index}/{len(annotations)}] {image_id}: {caption}")

        except Exception as e:
            print(f"[{index}/{len(annotations)}] Error {image_id}: {e}")

    print(f"\nFinished. Saved {len(rows)} captions to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()