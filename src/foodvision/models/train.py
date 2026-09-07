"""
FoodVision 1.0 - Model Training Pipeline
Fine-tunes a vision backbone (e.g. MobileNetV3 or EfficientNet) on the Food-101 / Kaggle dataset.
"""

import os
import argparse
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.models as models
from foodvision.config import config, DATA_DIR, MODELS_DIR
from foodvision.logger import logger
from foodvision.data.dataset import create_dataloaders


def build_model(num_classes: int = 101, architecture: str = "mobilenet_v3_small") -> nn.Module:
    """Instantiates a pretrained backbone with a new classification head."""
    logger.info("Building model architecture: %s for %d classes", architecture, num_classes)
    if architecture == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    elif architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    return model


def train(
    data_dir: Path,
    output_path: Path,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    architecture: str = "mobilenet_v3_small"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Starting training on device: %s", device)

    train_loader, val_loader, class_names = create_dataloaders(data_dir, batch_size=batch_size)
    if not train_loader:
        logger.error("Could not create dataloaders from %s. Please prepare dataset first.", data_dir)
        return

    num_classes = len(class_names)
    model = build_model(num_classes=num_classes, architecture=architecture)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == targets.data).item()
            total += targets.size(0)

        scheduler.step()
        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == targets.data).item()
                val_total += targets.size(0)

        val_acc = val_correct / max(1, val_total)
        elapsed = time.time() - start_time

        logger.info(
            "Epoch [%d/%d] (%0.1fs) - Train Loss: %0.4f | Train Acc: %0.2f%% | Val Acc: %0.2f%%",
            epoch, epochs, elapsed, train_loss, train_acc * 100, val_acc * 100
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model": model,
                "classes": class_names,
                "id2label": {i: name for i, name in enumerate(class_names)},
                "architecture": architecture,
                "val_acc": val_acc
            }, output_path)
            logger.info("Saved new best model checkpoint to %s (Val Acc: %0.2f%%)", output_path, val_acc * 100)

    logger.info("Training complete. Best Validation Accuracy: %0.2f%%", best_val_acc * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FoodVision DL Classifier")
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR / "food-101")
    parser.add_argument("--output_path", type=Path, default=MODELS_DIR / "food_classifier.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--arch", type=str, default="mobilenet_v3_small")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_path=args.output_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        architecture=args.arch
    )
