"""
FoodVision 1.0 Dataset Module
Provides PyTorch Dataset implementations, ImageFolder loaders,
and standard computer vision preprocessing pipelines for food classification.
"""

from pathlib import Path
from typing import Tuple, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from foodvision.config import config
from foodvision.logger import logger

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Standard training data augmentation pipeline."""
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """Deterministic validation and inference preprocessing pipeline."""
    return transforms.Compose([
        transforms.Resize((int(image_size * 1.14), int(image_size * 1.14))),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


class FoodImageFolderDataset(Dataset):
    """
    Standard PyTorch dataset parsing directory tree:
    root/
      <class_a>/img1.jpg, img2.jpg
      <class_b>/img1.jpg, img2.jpg
    """
    def __init__(self, root_dir: Path, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes: List[str] = []
        self.class_to_idx = {}
        self.samples: List[Tuple[Path, int]] = []

        self._scan_dataset()

    def _scan_dataset(self) -> None:
        if not self.root_dir.exists():
            logger.warning("Dataset root directory does not exist: %s", self.root_dir)
            return

        class_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        self.classes = sorted([d.name for d in class_dirs])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        for cls_name in self.classes:
            cls_folder = self.root_dir / cls_name
            cls_idx = self.class_to_idx[cls_name]
            for img_file in cls_folder.iterdir():
                if img_file.suffix.lower() in valid_exts:
                    self.samples.append((img_file, cls_idx))

        logger.info("Found %d images across %d classes in %s", len(self.samples), len(self.classes), self.root_dir)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, target = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error("Failed to load image %s: %s", img_path, e)
            # Return blank image placeholder
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, target


def create_dataloaders(
    data_dir: Path,
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[Optional[DataLoader], Optional[DataLoader], List[str]]:
    """Creates train and validation DataLoaders."""
    train_dir = data_dir / "train"
    val_dir = data_dir / "test"

    if not train_dir.exists():
        # Check if single images/ dir exists
        train_dir = data_dir / "images"
        val_dir = data_dir / "images"

    if not train_dir.exists():
        return None, None, []

    train_dataset = FoodImageFolderDataset(train_dir, transform=get_train_transforms())
    val_dataset = FoodImageFolderDataset(val_dir, transform=get_val_transforms())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, train_dataset.classes
