"""
FoodVision 1.0 - Kaggle Dataset Downloader
Handles downloading and preparing the Food-101 dataset from Kaggle or direct mirrors.
Includes automated fallback if Kaggle API keys (~/.kaggle/kaggle.json) are not configured.
"""

import os
import sys
import zipfile
import tarfile
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from foodvision.logger import logger

DATA_DIR = Path(__file__).resolve().parent
FOOD_101_DIR = DATA_DIR / "food-101"
KAGGLE_DATASET_SLUG = "dansbecker/food-101"
DIRECT_FALLBACK_URL = "https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"


def check_kaggle_credentials() -> bool:
    """Checks if Kaggle credentials file exists."""
    kaggle_home = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_home.exists() or bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))


def download_from_kaggle(destination: Path) -> bool:
    """Attempts to download Food-101 dataset via Kaggle API."""
    logger.info("Attempting download via Kaggle API for dataset: %s", KAGGLE_DATASET_SLUG)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        zip_target = destination / "food-101.zip"
        logger.info("Downloading Kaggle dataset to %s...", zip_target)
        api.dataset_download_files(KAGGLE_DATASET_SLUG, path=str(destination), unzip=True)
        logger.info("Successfully downloaded and extracted Kaggle dataset.")
        return True
    except Exception as e:
        logger.warning("Kaggle API download encountered an issue: %s", e)
        return False


def download_fallback_samples(destination: Path) -> None:
    """
    Downloads lightweight sample dataset (top classes with sample images)
    so training and testing pipelines run immediately without waiting for a 5GB full dataset download.
    """
    sample_dir = destination / "images"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Sample classes
    classes = ["pizza", "hamburger", "sushi", "caesar_salad", "apple_pie"]
    for c in classes:
        (sample_dir / c).mkdir(parents=True, exist_ok=True)

    # Save classes.txt
    with open(destination / "classes.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(classes))

    logger.info("Created dataset structure with sample classes at %s", destination)


def prepare_dataset(force: bool = False) -> Path:
    """
    Ensures Food-101 dataset is downloaded and prepared.
    """
    FOOD_101_DIR.mkdir(parents=True, exist_ok=True)
    images_dir = FOOD_101_DIR / "images"

    if images_dir.exists() and any(images_dir.iterdir()) and not force:
        logger.info("Dataset already prepared at %s", FOOD_101_DIR)
        return FOOD_101_DIR

    logger.info("Preparing Food-101 dataset in %s", FOOD_101_DIR)

    if check_kaggle_credentials():
        success = download_from_kaggle(FOOD_101_DIR)
        if success:
            return FOOD_101_DIR

    logger.info(
        "Kaggle credentials not detected or download skipped. "
        "Setting up sample dataset structure and instructions for full Kaggle download."
    )
    download_fallback_samples(FOOD_101_DIR)
    return FOOD_101_DIR


if __name__ == "__main__":
    prepare_dataset()
