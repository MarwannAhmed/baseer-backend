import math
import cv2
import numpy as np
import pytest
from app.features.color_detection.infer_color import extract_features, CHROMA_FILTER_THRESHOLD


def _solid(bgr_tuple, size=64):
    img = np.full((size, size, 3), bgr_tuple, dtype=np.uint8)
    return img


def test_extract_features_output_shape():
    img = _solid((0, 0, 200))  # red-ish
    features, *_ = extract_features(img)
    assert features.shape == (1, 5)


def test_extract_features_dtype():
    img = _solid((0, 0, 200))
    features, *_ = extract_features(img)
    assert features.dtype == np.float32


def test_extract_features_returns_six_values():
    img = _solid((0, 0, 200))
    result = extract_features(img)
    assert len(result) == 6  # features, L, A, B, chroma, hue


def test_extract_features_lab_ranges():
    img = _solid((120, 80, 40))
    _, L, A, B, chroma, hue = extract_features(img)
    assert 0.0 <= L <= 100.0
    assert 0.0 <= A <= 255.0
    assert 0.0 <= B <= 255.0


def test_extract_features_chroma_is_non_negative():
    img = _solid((100, 150, 200))
    _, _, _, _, chroma, _ = extract_features(img)
    assert chroma >= 0.0


def test_extract_features_hue_in_radians():
    img = _solid((0, 0, 255))  # red
    _, _, _, _, _, hue = extract_features(img)
    assert -math.pi <= hue <= math.pi


def test_extract_features_gray_image_falls_back():
    # Gray image has near-zero chroma — should fall back to all pixels without crashing
    img = _solid((128, 128, 128))
    features, *_ = extract_features(img)
    assert features.shape == (1, 5)


def test_extract_features_single_pixel_image():
    img = np.array([[[0, 128, 255]]], dtype=np.uint8)
    features, *_ = extract_features(img)
    assert features.shape == (1, 5)


def test_extract_features_pure_white():
    img = _solid((255, 255, 255))
    features, L, *_ = extract_features(img)
    assert L > 90  # white should have high L in LAB


def test_extract_features_pure_black():
    img = _solid((0, 0, 0))
    features, L, *_ = extract_features(img)
    assert L < 10  # black should have very low L in LAB
