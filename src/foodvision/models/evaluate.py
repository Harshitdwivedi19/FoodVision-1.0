"""
FoodVision 1.0 - Model Evaluation Pipeline
Calculates Top-1 accuracy, Top-5 accuracy, inference latency, and generates classification metrics.
"""

import time
import argparse
from pathlib import Path
from typing import Dict, Any
import torch
import torch.nn.functional as F
from foodvision.config import config, DATA_DIR, MODELS_DIR
from foodvision.logger import logger
from foodvision.data.dataset import create_dataloaders


def evaluate(model_path: Path, data_dir: Path, batch_size: int = 32) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Evaluating checkpoint: %s on %s", model_path, device)

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    model = checkpoint["model"]
    model.to(device)
    model.eval()

    _, val_loader, class_names = create_dataloaders(data_dir, batch_size=batch_size)
    if not val_loader:
        raise ValueError(f"No validation data found in {data_dir}")

    total_samples = 0
    top1_correct = 0
    top5_correct = 0
    latencies = []

    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            t0 = time.perf_counter()
            outputs = model(images)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) / images.size(0))

            probs = F.softmax(outputs, dim=-1)
            _, pred_top1 = torch.max(probs, 1)
            top1_correct += torch.sum(pred_top1 == targets.data).item()

            k = min(5, probs.size(1))
            _, pred_top5 = torch.topk(probs, k=k, dim=1)
            for i in range(targets.size(0)):
                if targets[i] in pred_top5[i]:
                    top5_correct += 1

            total_samples += targets.size(0)

    top1_acc = top1_correct / max(1, total_samples)
    top5_acc = top5_correct / max(1, total_samples)
    avg_latency_ms = (sum(latencies) / max(1, len(latencies))) * 1000

    results = {
        "total_samples": total_samples,
        "top1_accuracy": round(top1_acc * 100, 2),
        "top5_accuracy": round(top5_acc * 100, 2),
        "avg_latency_ms": round(avg_latency_ms, 2)
    }

    logger.info(
        "Evaluation Results: Top-1: %0.2f%% | Top-5: %0.2f%% | Latency: %0.2f ms/img",
        results["top1_accuracy"], results["top5_accuracy"], results["avg_latency_ms"]
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate FoodVision DL Classifier")
    parser.add_argument("--model_path", type=Path, default=MODELS_DIR / "food_classifier.pt")
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR / "food-101")
    args = parser.parse_args()

    evaluate(args.model_path, args.data_dir)
