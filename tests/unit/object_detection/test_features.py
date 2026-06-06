import numpy as np
import cv2

from app.features.object_detection.features import (
    canny_edge_map_feature,
    extract_features,
    hog_feature,
    sift_descriptor_feature,
)


def test_hog_feature_returns_float32_and_expected_length():
    patch = np.zeros((96, 96), dtype=np.uint8)
    patch[32:64, 32:64] = 255

    features = hog_feature(patch.astype(np.float32))

    assert features.dtype == np.float32
    assert features.shape[0] == 4356
    assert np.any(features > 0)


def test_sift_descriptor_feature_returns_128_length_and_normalized():
    patch = np.zeros((96, 96), dtype=np.uint8)
    patch[16:80, 16:80] = 255

    features = sift_descriptor_feature(patch.astype(np.float32))

    assert features.dtype == np.float32
    assert features.shape == (128,)
    assert np.all(features >= 0.0)
    assert np.all(features <= 1.0 + 1e-6)


def test_canny_edge_map_feature_returns_density_grid():
    patch = np.zeros((96, 96), dtype=np.uint8)
    cv2.line(patch, (0, 0), (95, 95), color=255, thickness=2)

    features = canny_edge_map_feature(patch.astype(np.float32))

    assert features.dtype == np.float32
    assert features.shape == (64,)
    assert np.all(features >= 0.0)
    assert np.all(features <= 1.0)
    assert np.any(features > 0.0)


def test_extract_features_accepts_color_image_and_returns_concatenated_vector():
    patch = np.zeros((120, 80, 3), dtype=np.uint8)
    patch[20:100, 10:70] = (255, 255, 255)

    features = extract_features(patch)

    assert features.dtype == np.float32
    assert features.ndim == 1
    assert features.shape[0] == 4548
    assert np.any(features > 0.0)
