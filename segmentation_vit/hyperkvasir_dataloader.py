
from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
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

        mask = torch.from_numpy(
            (np.array(mask) > 127).astype(np.int64)
        )

        return image, mask


def initialize_segmentation_dataloaders(
    data_dir=None,
    batch_size=16,
    image_size=224,
    num_workers=2,
    seed=42,
):
    if data_dir is None:
        data_dir = Path.home() / "mtl_data" / "segmented_polyp"

    data_dir = Path(data_dir)
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    if not masks_dir.is_dir():
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

    valid_extensions = {".jpg", ".jpeg", ".png"}

    image_files = {
        p.name: p
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in valid_extensions
    }

    mask_files = {
        p.name: p
        for p in masks_dir.iterdir()
        if p.is_file() and p.suffix.lower() in valid_extensions
    }

    missing_masks = set(image_files) - set(mask_files)
    missing_images = set(mask_files) - set(image_files)

    if missing_masks:
        raise FileNotFoundError(
            f"Missing masks for {len(missing_masks)} images: "
            f"{sorted(missing_masks)[:10]}"
        )

    if missing_images:
        raise FileNotFoundError(
            f"Missing images for {len(missing_images)} masks: "
            f"{sorted(missing_images)[:10]}"
        )

    samples = [
        (image_files[name], mask_files[name])
        for name in sorted(image_files)
    ]

    if len(samples) < 2:
        raise ValueError(
            f"Not enough image-mask pairs in {data_dir}"
        )

    train_samples, val_samples = train_test_split(
        samples,
        test_size=0.15,
        random_state=seed,
        shuffle=True,
    )

    train_loader = DataLoader(
        SegmentationDataset(
            train_samples,
            image_size=image_size,
            train=True,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        SegmentationDataset(
            val_samples,
            image_size=image_size,
            train=False,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"[HyperKvasir Segmentation] Total pairs: {len(samples)}")
    print(f"[HyperKvasir Segmentation] Training images: {len(train_samples)}")
    print(f"[HyperKvasir Segmentation] Validation images: {len(val_samples)}")

    return train_loader, val_loader
