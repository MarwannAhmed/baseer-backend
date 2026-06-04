"""
features.py — Combined Feature Extraction
==========================================
Implements the three-part feature descriptor used by this detector:

  1. HOG  (Histogram of Oriented Gradients)
  2. Dense SIFT-inspired gradient histograms
  3. Canny edge density map

Final feature vector = concat(HOG, SIFT-inspired, Canny)
"""

import numpy as np
import cv2
from skimage.feature import hog

from app.features.object_detection import config


def _to_gray(patch):
    """Convert BGR or grayscale patch to float32 grayscale."""
    if len(patch.shape) == 3 and patch.shape[2] == 3:
        return cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return patch.astype(np.float32)


def extract_hog_features(patch):
    gray = _to_gray(patch)
    gray = np.sqrt(gray / 255.0)
    features = hog(
        gray,
        orientations=config.HOG_ORIENTATIONS,
        pixels_per_cell=config.HOG_PIXELS_PER_CELL,
        cells_per_block=config.HOG_CELLS_PER_BLOCK,
        block_norm=config.HOG_BLOCK_NORM,
        feature_vector=True,
        channel_axis=None,
    )
    return features.astype(np.float32)


def extract_sift_inspired_features(patch):
    gray      = _to_gray(patch)
    gx        = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
    gy        = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    angle     = np.arctan2(np.abs(gy), np.abs(gx)) * (180.0 / np.pi)

    h, w = gray.shape
    gs   = config.SIFT_GRID_SIZE
    nb   = config.SIFT_BINS
    ch   = h // gs
    cw   = w // gs

    features = []
    for i in range(gs):
        for j in range(gs):
            y1, y2     = i * ch, (i + 1) * ch
            x1, x2     = j * cw, (j + 1) * cw
            cell_mag   = magnitude[y1:y2, x1:x2]
            cell_angle = angle[y1:y2, x1:x2]
            gauss      = cv2.getGaussianKernel(ch, ch / 2.0) * \
                         cv2.getGaussianKernel(cw, cw / 2.0).T
            gauss      = gauss / (gauss.max() + 1e-8)
            cell_mag   = cell_mag * gauss
            hist, _    = np.histogram(
                cell_angle.ravel(), bins=nb, range=(0.0, 90.0),
                weights=cell_mag.ravel(),
            )
            norm = np.linalg.norm(hist) + 1e-8
            hist = hist / norm
            hist = np.clip(hist, 0.0, 0.2)
            hist = hist / (np.linalg.norm(hist) + 1e-8)
            features.extend(hist)

    return np.array(features, dtype=np.float32)


def extract_canny_features(patch):
    gray  = _to_gray(patch).astype(np.uint8)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    gs    = config.CANNY_GRID_SIZE
    h, w  = edges.shape
    ch    = h // gs
    cw    = w // gs
    features = []
    for i in range(gs):
        for j in range(gs):
            cell    = edges[i*ch:(i+1)*ch, j*cw:(j+1)*cw]
            density = float(np.mean(cell > 0))
            features.append(density)
    return np.array(features, dtype=np.float32)


def extract_features(patch):
    """
    Extract the full combined feature vector from one image patch.
    The patch is resized to WINDOW_SIZE before extraction.
    """
    resized    = cv2.resize(patch, config.WINDOW_SIZE)
    hog_feat   = extract_hog_features(resized)
    sift_feat  = extract_sift_inspired_features(resized)
    canny_feat = extract_canny_features(resized)
    return np.concatenate([hog_feat, sift_feat, canny_feat]).astype(np.float32)


def get_feature_dim():
    dummy = np.zeros((*config.WINDOW_SIZE[::-1], 3), dtype=np.uint8)
    return extract_features(dummy).shape[0]