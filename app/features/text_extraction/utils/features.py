import cv2
import numpy as np

from app.features.text_extraction.config import parameters


def normalise(char_img):
    h, w = char_img.shape
    if h == 0 or w == 0:
        return np.ones((parameters["features"]["norm_size"], parameters["features"]["norm_size"]), dtype=np.uint8) * 255

    scale  = (parameters["features"]["norm_size"] - 4) / max(h, w)
    new_h  = max(1, int(h * scale))
    new_w  = max(1, int(w * scale))
    scaled = cv2.resize(char_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.ones((parameters["features"]["norm_size"], parameters["features"]["norm_size"]), dtype=np.uint8) * 255
    y_off  = (parameters["features"]["norm_size"] - new_h) // 2
    x_off  = (parameters["features"]["norm_size"] - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = scaled

    return canvas

def skeletonise(norm_img):
    inv = cv2.bitwise_not(norm_img)
    return cv2.ximgproc.thinning(inv, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

def _outline_features(norm_img):
    inv = (norm_img < 128).astype(np.uint8) * 255
    contours, _ = cv2.findContours(inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return np.zeros(parameters["features"]["outline_samples"] * 3 + parameters["features"]["outline_samples"] * 3, dtype=np.float32)

    outer   = max(contours, key=lambda c: len(c))
    inners  = [c for c in contours if cv2.contourArea(c) < cv2.contourArea(outer) and len(c) >= 2]
    inner   = max(inners, key=lambda c: len(c)) if inners else None

    def sample_contour(contour):
        total   = len(contour)
        indices = np.linspace(0, total - 1, parameters["features"]["outline_samples"], dtype=int)
        sampled = contour[indices].reshape(-1, 2).astype(np.float32)
        nx      = sampled[:, 0] / (parameters["features"]["norm_size"] - 1)
        ny      = sampled[:, 1] / (parameters["features"]["norm_size"] - 1)
        next_idx = (indices + 1) % total
        nextp    = contour[next_idx].reshape(-1, 2).astype(np.float32)
        dx       = nextp[:, 0] - sampled[:, 0]
        dy       = nextp[:, 1] - sampled[:, 1]
        angles   = np.arctan2(dy, dx)
        dir_norm = (angles + np.pi) / (2 * np.pi)
        return np.concatenate([nx, ny, dir_norm]).astype(np.float32)

    outer_feat = sample_contour(outer)
    inner_feat = sample_contour(inner) if inner is not None else np.zeros(parameters["features"]["outline_samples"] * 3, dtype=np.float32)

    return np.concatenate([outer_feat, inner_feat]).astype(np.float32)

def _grid_features(norm_img):
    inv      = (norm_img < 128).astype(np.float32)
    cell_h   = parameters["features"]["norm_size"] // parameters["features"]["grid_rows"]
    cell_w   = parameters["features"]["norm_size"] // parameters["features"]["grid_cols"]
    features = np.zeros(parameters["features"]["grid_rows"] * parameters["features"]["grid_cols"], dtype=np.float32)

    for r in range(parameters["features"]["grid_rows"]):
        for c in range(parameters["features"]["grid_cols"]):
            cell = inv[r * cell_h:(r + 1) * cell_h,
                       c * cell_w:(c + 1) * cell_w]
            features[r * parameters["features"]["grid_cols"] + c] = float(cell.mean())

    return features

def _direction_histogram(skel_img):
    sobelx    = cv2.Sobel(skel_img, cv2.CV_32F, 1, 0, ksize=3)
    sobely    = cv2.Sobel(skel_img, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    angles    = np.arctan2(sobely, sobelx)

    mask      = magnitude >= 5.0
    flat_a    = angles[mask]
    flat_m    = magnitude[mask]

    bins      = np.linspace(-np.pi, np.pi, parameters["features"]["direction_bins"] + 1)
    hist, _   = np.histogram(flat_a, bins=bins, weights=flat_m)
    total     = hist.sum()
    if total > 0:
        hist = hist / total

    return hist.astype(np.float32)

def _projection_profiles(norm_img):
    inv          = (norm_img < 128).astype(np.float32)
    h_profile    = inv.sum(axis=1).astype(np.float32)
    v_profile    = inv.sum(axis=0).astype(np.float32)

    h_resampled  = cv2.resize(h_profile.reshape(1, -1), (parameters["features"]["profile_bins"], 1), interpolation=cv2.INTER_AREA).flatten()
    v_resampled  = cv2.resize(v_profile.reshape(1, -1), (parameters["features"]["profile_bins"], 1), interpolation=cv2.INTER_AREA).flatten()

    h_norm       = h_resampled / (h_resampled.max() + 1e-8)
    v_norm       = v_resampled / (v_resampled.max() + 1e-8)

    return np.concatenate([h_norm, v_norm]).astype(np.float32)

def _topology_features(norm_img):
    inv          = (norm_img < 128).astype(np.uint8) * 255
    contours, _  = cv2.findContours(inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    n_holes      = max(0, len(contours) - 1)

    euler        = cv2.morphologyEx(inv, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    euler_norm   = float(euler.sum()) / (parameters["features"]["norm_size"] * parameters["features"]["norm_size"] * 255.0)

    return np.array([min(n_holes, 5) / 5.0, euler_norm], dtype=np.float32)

def extract(char_img):
    norm     = normalise(char_img)
    skel     = skeletonise(norm)
    skel_inv = cv2.bitwise_not(skel)

    outline  = _outline_features(norm)
    grid     = _grid_features(skel_inv)
    dir_hist = _direction_histogram(skel_inv)
    profiles = _projection_profiles(skel_inv)
    topology = _topology_features(norm)

    return np.concatenate([outline, grid, dir_hist, profiles, topology]).astype(np.float32)

def extract_batch(char_imgs):
    return np.stack([extract(img) for img in char_imgs], axis=0)