import numpy as np
import pytest

import app.features.object_detection.predict as predict


class FakeModel:
    def predict_proba(self, _):
        return np.array([[0.1, 0.9]], dtype=np.float32)


class DummySelectiveSearch:
    def setBaseImage(self, image):
        self.image = image

    def switchToSelectiveSearchFast(self):
        return None

    def process(self):
        return [(0, 0, 50, 50)]


def test_run_detection_formats_output_and_handles_errors(monkeypatch):
    monkeypatch.setattr(
        predict, "decode_image", lambda data: np.zeros((96, 96, 3), dtype=np.uint8)
    )
    monkeypatch.setattr(predict, "resize_if_large", lambda image: image)
    monkeypatch.setattr(
        predict, "detect_objects", lambda image: ([[0, 0, 50, 50]], [0.95], [0])
    )
    monkeypatch.setattr(predict, "class_labels", ["person"], raising=False)

    result = predict.run_detection(b"dummy")

    assert result == [
        {
            "label": "person",
            "confidence": 0.95,
            "bbox": {"x1": 0, "y1": 0, "x2": 50, "y2": 50},
        }
    ]


def test_run_detection_returns_empty_when_decode_fails(monkeypatch):
    monkeypatch.setattr(predict, "decode_image", lambda data: None)
    result = predict.run_detection(b"dummy")

    assert result == []
