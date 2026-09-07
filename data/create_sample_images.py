"""
Generates synthetic sample images for unit testing and offline demo testing.
"""

from pathlib import Path
from PIL import Image, ImageDraw

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def create_sample_food_image(filename: str, label: str, bg_color: tuple, accent_color: tuple):
    img = Image.new("RGB", (300, 300), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Draw plate circle
    draw.ellipse([30, 30, 270, 270], fill=(245, 245, 245), outline=(200, 200, 200), width=4)
    # Draw food body
    draw.ellipse([60, 60, 240, 240], fill=accent_color)
    # Draw accents
    draw.ellipse([90, 90, 130, 130], fill=(220, 50, 50))
    draw.ellipse([160, 140, 200, 180], fill=(50, 180, 50))

    img_path = SAMPLES_DIR / filename
    img.save(img_path, format="JPEG", quality=90)
    print(f"Generated sample image: {img_path}")


def generate_all():
    create_sample_food_image("sample_pizza.jpg", "Pizza", (30, 41, 59), (230, 160, 50))
    create_sample_food_image("sample_salad.jpg", "Caesar Salad", (30, 41, 59), (70, 180, 80))
    create_sample_food_image("sample_sushi.jpg", "Sushi", (30, 41, 59), (230, 90, 70))


if __name__ == "__main__":
    generate_all()
