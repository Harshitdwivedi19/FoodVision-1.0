"""
Unit tests for FoodVision 1.0 Food Classifier
"""

import pytest
from PIL import Image
from foodvision.models.classifier import classifier, Prediction


def test_classifier_predict_synthetic_image():
    # Create a 224x224 RGB image
    img = Image.new("RGB", (224, 224), color=(200, 100, 50))
    predictions = classifier.predict(img, top_k=3)

    assert isinstance(predictions, list)
    assert len(predictions) == 3
    assert isinstance(predictions[0], Prediction)
    assert predictions[0].confidence >= 0.0
    assert predictions[0].rank == 1
    assert len(predictions[0].label) > 0


def test_classifier_accepts_bytes():
    import io
    img = Image.new("RGB", (100, 100), color=(50, 150, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    predictions = classifier.predict(img_bytes, top_k=2)
    assert len(predictions) == 2
    assert predictions[0].confidence >= 0.0
