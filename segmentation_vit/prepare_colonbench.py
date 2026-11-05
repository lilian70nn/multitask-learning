
from pathlib import Path
import csv
import os
import cv2
import numpy as np
from PIL import Image

RAW_DIR = Path("/content/drive/MyDrive/colonbench_raw")
OUTPUT_DIR = Path("/content/drive/MyDrive/datasets/colonbench_segmentation")
MANIFEST_PATH = OUTPUT_DIR / "manifest.csv"
COMPLETE_FLAG = OUTPUT_DIR / ".complete"


def prepare_colonbench():
    if COMPLETE_FLAG.is_file() and MANIFEST_PATH.is_file():
        print("[ColonBench] Prepared data already exists.")
        return OUTPUT_DIR

    video_dir = RAW_DIR / "videos"
    source_mask_dir = RAW_DIR / "masks"

    if not video_dir.is_dir(): raise FileNotFoundError(f"Video directory not found: {video_dir}")
    if not source_mask_dir.is_dir(): raise FileNotFoundError(f"Mask directory not found: {source_mask_dir}")

    videos = sorted(video_dir.rglob("*.mp4"))
    if not videos: raise ValueError(f"No MP4 videos found in {video_dir}")

    (OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "masks").mkdir(parents=True, exist_ok=True)

    records = []

    for video_number, video_path in enumerate(videos, 1):
        video_id = video_path.name
        video_stem = video_path.stem
        source_masks = sorted((source_mask_dir / video_stem).glob("*.png"))

        if not source_masks:
            raise FileNotFoundError(f"No masks found for video: {video_id}")

        print(f"[ColonBench] Video {video_number}/{len(videos)}: {video_id} ({len(source_masks)} masks)")
        image_dir = OUTPUT_DIR / "images" / video_stem
        mask_dir = OUTPUT_DIR / "masks" / video_stem
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened(): raise RuntimeError(f"Cannot open video: {video_path}")

        try:
            for source_mask in source_masks:
                if source_mask.stat().st_size == 0:
                    print(f"[ColonBench] Skipping empty mask: {source_mask}")
                    continue
                frame_index = int(source_mask.stem)
                image_path = image_dir / f"{frame_index:06d}.jpg"
                mask_path = mask_dir / f"{frame_index:06d}.png"

                if not image_path.is_file() or not mask_path.is_file():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    success, frame = cap.read()
                    if not success: raise RuntimeError(f"Cannot read frame {frame_index} from {video_id}")

                    with Image.open(source_mask) as m:
                        mask = np.asarray(m.convert("L")).copy()

                    if frame.shape[:2] != mask.shape:
                        raise ValueError(f"Image/mask size mismatch: {video_id}, frame {frame_index}, image={frame.shape[:2]}, mask={mask.shape}")

                    if not np.isin(np.unique(mask), [0, 255]).all():
                        raise ValueError(f"Unexpected mask pixel values: {source_mask}")

                    if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                        raise RuntimeError(f"Cannot save image: {image_path}")

                    Image.fromarray(mask).save(mask_path)

                records.append({
                    "video_id": video_id,
                    "frame_index": frame_index,
                    "description": "",
                    "image_path": str(image_path),
                    "mask_path": str(mask_path)
                })
        finally:
            cap.release()

    if not records: raise RuntimeError("No image-mask pairs were prepared.")

    temp_manifest = MANIFEST_PATH.with_suffix(".csv.tmp")
    with open(temp_manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "frame_index", "description", "image_path", "mask_path"])
        writer.writeheader()
        writer.writerows(records)

    os.replace(temp_manifest, MANIFEST_PATH)
    COMPLETE_FLAG.write_text(str(len(records)))

    print(f"[ColonBench] Preparation completed: {len(records)} image-mask pairs.")
    return OUTPUT_DIR
