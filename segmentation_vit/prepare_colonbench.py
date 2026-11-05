from pathlib import Path
import csv
import os
import cv2
import numpy as np
import shutil
import tempfile
from PIL import Image
from collections import Counter
from tqdm.auto import tqdm

RAW_DIR = Path("/content/drive/MyDrive/colonbench_raw")
OUTPUT_DIR = Path("/content/drive/MyDrive/colonbench_segmentation")
MANIFEST_PATH = OUTPUT_DIR / "manifest.csv"
COMPLETE_FLAG = OUTPUT_DIR / ".complete"
SKIPPED_PATH = OUTPUT_DIR / "skipped_frames.csv"


def prepare_colonbench():
    video_dir = RAW_DIR / "videos"
    source_mask_dir = RAW_DIR / "masks"

    if not video_dir.is_dir(): raise FileNotFoundError(f"Video directory not found: {video_dir}")
    if not source_mask_dir.is_dir(): raise FileNotFoundError(f"Mask directory not found: {source_mask_dir}")

    videos = sorted(video_dir.rglob("*.mp4"))
    if not videos: raise ValueError(f"No MP4 videos found in {video_dir}")

    (OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "masks").mkdir(parents=True, exist_ok=True)

    COMPLETE_FLAG.unlink(missing_ok=True)
    records = []
    skipped = []
    skipped_counts = Counter()

    def record_skip(video_id, frame_index, reason, detail):
        skipped.append({
            "video_id": video_id,
            "frame_index": frame_index,
            "reason": reason,
            "detail": detail
        })
        skipped_counts[reason] += 1
        print(f"[ColonBench] Skipping: {video_id}, frame {frame_index}, {reason}: {detail}")

    for video_number, video_path in enumerate(tqdm(videos, desc="Processing Videos", unit="video"), 1):
        video_id = video_path.name
        video_stem = video_path.stem
        source_masks = sorted((source_mask_dir / video_stem).glob("*.png"))

        if not source_masks:
            continue

        print(f"[ColonBench] Video {video_number}/{len(videos)}: {video_id} ({len(source_masks)} masks)")

        image_dir = OUTPUT_DIR / "images" / video_stem
        mask_dir = OUTPUT_DIR / "masks" / video_stem
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="colonbench_", dir="/content") as temp:
            temp_dir = Path(temp)
            local_video = temp_dir / video_id
            local_images = temp_dir / "images"
            local_masks = temp_dir / "masks"
            local_images.mkdir()
            local_masks.mkdir()

            shutil.copy2(video_path, local_video)

            cap = cv2.VideoCapture(str(local_video))
            if not cap.isOpened(): raise RuntimeError(f"Cannot open video: {local_video}")

            video_records = []

            try:
                for source_mask in tqdm(source_masks, desc=f"Frames {video_stem[:8]}", unit="frame", leave=False):
                    frame_index = int(source_mask.stem)
                    if source_mask.stat().st_size == 0:
                        record_skip(video_id, frame_index, "empty_mask", str(source_mask))
                        continue

                    image_name = f"{frame_index:06d}.jpg"
                    mask_name = f"{frame_index:06d}.png"

                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    success, frame = cap.read()
                    if not success:
                        record_skip(
                            video_id, frame_index,
                            "frame_read_failed",
                            "Cannot read video frame"
                        )
                        continue

                    with Image.open(source_mask) as m:
                        mask = np.asarray(m.convert("L")).copy()

                    if frame.shape[:2] != mask.shape:
                        if (
                            mask.shape[0] == frame.shape[0] + 1
                            and mask.shape[1] == frame.shape[1]
                            and np.all(mask[-1] == 0)
                        ):
                            mask = mask[:-1, :].copy()
                        else:
                            record_skip(
                                video_id, frame_index, "size_mismatch",
                                f"image={frame.shape[:2]}, mask={mask.shape}"
                            )
                            continue

                    if not np.isin(np.unique(mask), [0, 255]).all():
                        record_skip(
                            video_id, frame_index,
                            "invalid_mask_values",
                            str(source_mask)
                        )
                        continue

                    if not cv2.imwrite(str(local_images / image_name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                        raise RuntimeError(f"Cannot save image: {local_images / image_name}")

                    Image.fromarray(mask).save(local_masks / mask_name)

                    video_records.append({
                        "video_id": video_id,
                        "frame_index": frame_index,
                        "description": "",
                        "image_path": str(image_dir / image_name),
                        "mask_path": str(mask_dir / mask_name)
                    })

            finally:
                cap.release()

            for record in tqdm(video_records, desc=f"Saving {video_stem[:8]}", unit="frame", leave=False):
                image_path = Path(record["image_path"])
                mask_path = Path(record["mask_path"])

                shutil.copy2(local_images / image_path.name, image_path)
                shutil.copy2(local_masks / mask_path.name, mask_path)

            records.extend(video_records)

    if not records:
        raise RuntimeError("No image-mask pairs were prepared.")

    temp_manifest = MANIFEST_PATH.with_suffix(".csv.tmp")

    with open(temp_manifest, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["video_id", "frame_index", "description", "image_path", "mask_path"]
        )
        writer.writeheader()
        writer.writerows(records)

    os.replace(temp_manifest, MANIFEST_PATH)

    with open(SKIPPED_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["video_id", "frame_index", "reason", "detail"]
        )
        writer.writeheader()
        writer.writerows(skipped)


    COMPLETE_FLAG.write_text(str(len(records)))

    print(f"[ColonBench] Preparation completed: {len(records)} image-mask pairs.")
    print(f"[ColonBench] Skipped frames: {len(skipped)}")

    for reason, count in sorted(skipped_counts.items()):
        print(f"  {reason}: {count}")

    print(f"[ColonBench] Skipped report: {SKIPPED_PATH}")

    return OUTPUT_DIR