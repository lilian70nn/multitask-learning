from pathlib import Path
import random
import csv

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from collections import Counter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode



class SegmentationDataset(Dataset):
    def __init__(self, samples, image_size=224, train=False):
        self.samples = samples
        self.image_size = image_size
        self.train = train

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        with Image.open(mask_path) as m:
            mask = m.convert("L")

        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
        )

        mask = TF.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST,
        )

        if self.train:
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if random.random() < 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

        image = TF.to_tensor(image)
        image = self.normalize(image)
        mask = torch.from_numpy((np.array(mask) > 127).astype(np.int64))

        return image, mask


def initialize_segmentation_dataloaders(
    data_dir=Path.home() / "mtl_data" / "colonbench_segmentation",
    batch_size=16,
    image_size=224,
    num_workers=2,
    seed=42,
):
    
    data_dir = Path(data_dir)
    manifest_path = data_dir / "manifest.csv"
    complete_flag = data_dir / ".complete"

    if not manifest_path.is_file() or not complete_flag.is_file():
        raise RuntimeError(f"ColonBench preparation is missing or incomplete: {data_dir}")

    with open(manifest_path, newline="") as f:
        records = list(csv.DictReader(f))

    if len(records) < 2:
        raise ValueError(f"Not enough image-mask pairs in {manifest_path}")

    samples = []
    groups = []

    for row in records:
        image_path, mask_path = Path(row["image_path"]), Path(row["mask_path"])
        if not image_path.is_file(): raise FileNotFoundError(f"Image not found: {image_path}")
        if not mask_path.is_file(): raise FileNotFoundError(f"Mask not found: {mask_path}")
        samples.append((image_path, mask_path))
        groups.append(row["video_id"])

    if len(set(groups)) < 2:
        raise ValueError("At least two different videos are required for train/validation splitting.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    train_indices, val_indices = next(splitter.split(samples, groups=groups))

    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]
    train_videos = {groups[i] for i in train_indices}
    val_videos = {groups[i] for i in val_indices}

    assert train_videos.isdisjoint(val_videos), "Video leakage detected between training and validation."

    # Count the number of frames in each training video
    train_video_ids = [groups[i] for i in train_indices]
    video_counts = Counter(train_video_ids)

    # Each video receives approximately equal sampling probability
    sample_weights = torch.DoubleTensor([
        1.0 / video_counts[video_id]
        for video_id in train_video_ids
    ])

    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_samples),
        replacement=True,
    )

    train_loader = DataLoader(
        SegmentationDataset(train_samples, image_size=image_size, train=True),
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        SegmentationDataset(val_samples, image_size=image_size, train=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"[Segmentation] Total image-mask pairs: {len(samples)}")
    print(f"[Segmentation] Total videos: {len(set(groups))}")
    print(f"[Segmentation] Training videos: {len(train_videos)} | Training images: {len(train_samples)}")
    print(f"[Segmentation] Validation videos: {len(val_videos)} | Validation images: {len(val_samples)}")

    return train_loader, val_loader
