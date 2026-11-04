
from pathlib import Path
import csv
import os
import shutil
import cv2
import numpy as np
from PIL import Image
from datasets import load_dataset
from huggingface_hub import hf_hub_download


REPO_ID = "ajhamdi/colon-bench"
RAW_DIR = Path("/content/colonbench_raw")
OUTPUT_DIR = Path("/content/colonbench_segmentation")
MANIFEST_PATH = OUTPUT_DIR / "manifest.csv"
COMPLETE_FLAG = OUTPUT_DIR / ".complete"


def download_file(repo_path):
    local_path = RAW_DIR / repo_path
    if local_path.is_file(): return local_path

    local_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(repo_id=REPO_ID, filename=repo_path, repo_type="dataset")
    shutil.copy2(downloaded, local_path)
    return local_path


def prepare_colonbench():
    if COMPLETE_FLAG.is_file() and MANIFEST_PATH.is_file():
        print("[ColonBench] Prepared data already exists.")
        return OUTPUT_DIR

    print("[ColonBench] Loading segmentation metadata...")
    seg = load_dataset(REPO_ID, "segmentation", split="test")

    (OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "masks").mkdir(parents=True, exist_ok=True)

    records = []
    expected_pairs = sum(int(sample["num_masks"]) for sample in seg)

    for video_number, sample in enumerate(seg, 1):
        video_id = sample["video_id"]
        video_stem = Path(video_id).stem
        frame_indices = sample["mask_frame_indices"]
        mask_paths = sample["mask_paths"]

        if len(frame_indices) != len(mask_paths) or len(frame_indices) != sample["num_masks"]:
            raise ValueError(f"Inconsistent metadata: {video_id}")

        print(f"[ColonBench] Video {video_number}/{len(seg)}: {video_id} ({len(frame_indices)} masks)")
        video_path = download_file(f"videos/{video_id}")
        image_dir = OUTPUT_DIR / "images" / video_stem
        mask_dir = OUTPUT_DIR / "masks" / video_stem
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened(): raise RuntimeError(f"Cannot open video: {video_path}")

        try:
            for frame_index, mask_relative_path in sorted(zip(frame_indices, mask_paths)):
                frame_index = int(frame_index)
                image_path = image_dir / f"{frame_index:06d}.jpg"
                mask_path = mask_dir / f"{frame_index:06d}.png"

                if not image_path.is_file() or not mask_path.is_file():
                    source_mask = download_file(mask_relative_path)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    success, frame = cap.read()
                    if not success: raise RuntimeError(f"Cannot read frame {frame_index} from {video_id}")

                    with Image.open(source_mask) as m:
                        mask = np.asarray(m.convert("L")).copy()

                    if frame.shape[:2] != mask.shape:
                        raise ValueError(f"Image/mask size mismatch: {video_id}, frame {frame_index}")

                    if not np.isin(np.unique(mask), [0, 255]).all():
                        raise ValueError(f"Unexpected mask pixel values: {mask_relative_path}")

                    if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                        raise RuntimeError(f"Cannot save image: {image_path}")

                    Image.fromarray(mask).save(mask_path)

                records.append({
                    "video_id": video_id,
                    "frame_index": frame_index,
                    "description": sample["description"],
                    "image_path": str(image_path),
                    "mask_path": str(mask_path)
                })
        finally:
            cap.release()

    if len(records) != expected_pairs:
        raise RuntimeError(f"Expected {expected_pairs} pairs, prepared {len(records)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temp_manifest = MANIFEST_PATH.with_suffix(".csv.tmp")

    with open(temp_manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "frame_index", "description", "image_path", "mask_path"])
        writer.writeheader()
        writer.writerows(records)

    os.replace(temp_manifest, MANIFEST_PATH)
    COMPLETE_FLAG.write_text(str(len(records)))

    print(f"[ColonBench] Preparation completed: {len(records)} image-mask pairs.")
    return OUTPUT_DIR
