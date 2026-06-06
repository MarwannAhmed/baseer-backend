#!/usr/bin/env python3
"""
train_and_eval.py
────────────────────────────────────────────────────────────────────────────────
1. Loads your dataset (from build_dataset.py output)
2. Extracts dominant LAB color from every image — same pipeline as color_detector.dart
3. Evaluates your CURRENT rule-based algorithm → baseline accuracy + confusion matrix
4. Trains an SVM on the same features → improved accuracy + confusion matrix
5. Saves the trained model as color_svm.pkl

All constants and the classify_rules() function are in exact sync with
the CURRENT color_detector.dart (hue-bucketing classifier, clamped gray world,
chroma_filter=10, neutral_chroma=15, white_L=210, brown rule).

Requirements:
  pip install opencv-python scikit-learn matplotlib numpy

Usage:
  python train_and_eval.py --dataset ./dataset
  python train_and_eval.py --dataset ./dataset --output ./results
────────────────────────────────────────────────────────────────────────────────
"""

import cv2
import csv
import math
import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.svm            import SVC
from sklearn.preprocessing   import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics         import (accuracy_score, classification_report,
                                     confusion_matrix, ConfusionMatrixDisplay)
from sklearn.cluster         import KMeans

# ── Constants — must match color_detector.dart EXACTLY ───────────────────────

GW_CLAMP_MIN             = 0.75    # _gwClampMin
GW_CLAMP_MAX             = 1.50    # _gwClampMax
CHROMA_FILTER_THRESHOLD  = 10.0   # _chromaFilterThreshold
NEUTRAL_CHROMA_THRESHOLD = 15.0   # _neutralChromaThreshold
WHITE_L_THRESHOLD        = 210.0  # _whiteLThreshold  (OpenCV L, 0-255)
BLACK_L_THRESHOLD        = 50.0   # _blackLThreshold
KMEANS_K                 = 3      # _kmeansK
KMEANS_MAX_ITER          = 20     # _kmeansMaxIter
PIXEL_STRIDE             = 3      # _pixelStride

# Brown rule (_brownHueMax, _brownChromaMax, _brownLMax)
BROWN_HUE_MAX    = 83.0
BROWN_CHROMA_MAX = 45.0
BROWN_L_MAX      = 145.0

# Pink/Red lightness split (_pinkLMin)
PINK_L_MIN = 160.0

# Hue-angle bucket boundaries in degrees [0, 360)
H_PINK_RED      = 16.0   # _hPinkRed
H_RED_ORANGE    = 51.0   # _hRedOrange
H_ORANGE_YELLOW = 84.0   # _hOrangeYellow
H_YELLOW_GREEN  = 119.0  # _hYellowGreen
H_GREEN_BLUE    = 220.0  # _hGreenBlue
H_BLUE_PURPLE   = 315.0  # _hBluePurple
H_PURPLE_PINK   = 340.0  # _hPurplePink

COLOR_ORDER = ['red', 'orange', 'yellow', 'green', 'blue',
               'purple', 'pink', 'brown', 'white', 'gray', 'black']

# ── Gray-world AWB with clamping (matches _buildLabFrame) ────────────────────

def gray_world_awb(bgr: np.ndarray) -> np.ndarray:
    """
    Clamped gray-world AWB — mirrors _buildLabFrame in color_detector.dart.
    Scale factors are clamped to [GW_CLAMP_MIN, GW_CLAMP_MAX] to prevent
    dominant-colour scenes from crushing the hue signal.
    Input/output: BGR uint8.
    """
    r = bgr[:, :, 2].astype(np.float64)
    g = bgr[:, :, 1].astype(np.float64)
    b = bgr[:, :, 0].astype(np.float64)

    eps      = 1e-6
    r_mean   = r.mean(); g_mean = g.mean(); b_mean = b.mean()
    gray_avg = (r_mean + g_mean + b_mean) / 3.0

    r_scale = float(np.clip(gray_avg / (r_mean + eps), GW_CLAMP_MIN, GW_CLAMP_MAX))
    g_scale = float(np.clip(gray_avg / (g_mean + eps), GW_CLAMP_MIN, GW_CLAMP_MAX))
    b_scale = float(np.clip(gray_avg / (b_mean + eps), GW_CLAMP_MIN, GW_CLAMP_MAX))

    out = bgr.astype(np.float64)
    out[:, :, 2] = np.clip(out[:, :, 2] * r_scale, 0, 255)
    out[:, :, 1] = np.clip(out[:, :, 1] * g_scale, 0, 255)
    out[:, :, 0] = np.clip(out[:, :, 0] * b_scale, 0, 255)
    return out.astype(np.uint8)

# ── Dominant LAB extraction — mirrors the full Dart pipeline ─────────────────

def extract_dominant_lab(bgr_crop: np.ndarray):
    """
    Runs the color_detector.dart pipeline on a BGR crop.
    Returns (L, A, B) in OpenCV scale (0-255).

    AWB is intentionally skipped — applying it to a single-color crop pushes
    the dominant hue toward gray, destroying the color signal.
    """
    lab    = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    pixels = lab[::PIXEL_STRIDE, ::PIXEL_STRIDE].reshape(-1, 3)

    a_c    = pixels[:, 1] - 128.0
    b_c    = pixels[:, 2] - 128.0
    chroma = np.sqrt(a_c ** 2 + b_c ** 2)
    saturated = pixels[chroma >= CHROMA_FILTER_THRESHOLD]
    if len(saturated) < 10:
        saturated = pixels

    k = min(KMEANS_K, len(saturated))
    if k < 2:
        dominant = saturated.mean(axis=0)
    else:
        km = KMeans(n_clusters=k, init='k-means++', n_init=1,
                    max_iter=KMEANS_MAX_ITER, random_state=0)
        km.fit(saturated)
        counts   = np.bincount(km.labels_)
        dominant = km.cluster_centers_[counts.argmax()]

    return float(dominant[0]), float(dominant[1]), float(dominant[2])

# ── Rule-based classifier — exact mirror of _labToColorName in Dart ───────────
#
# Step-by-step:
#   1. Neutral guard  — chroma < NEUTRAL_CHROMA_THRESHOLD → white/gray/black
#   2. Hue angle      — atan2(B-128, A-128) → [0, 360)
#   3. Brown rule     — warm hue + low chroma + moderate L
#   4. Hue bucketing  — midpoints between LAB prototype anchors
#   5. Pink/Red split — same hue zone, split by lightness

def classify_rules(L: float, rawA: float, rawB: float) -> str:
    A      = rawA - 128.0
    B      = rawB - 128.0
    chroma = math.sqrt(A * A + B * B)

    # 1. Neutral guard
    if chroma < NEUTRAL_CHROMA_THRESHOLD:
        if L > WHITE_L_THRESHOLD: return 'white'
        if L < BLACK_L_THRESHOLD: return 'black'
        return 'gray'

    # 2. Hue angle in [0, 360)
    h = math.degrees(math.atan2(B, A))
    if h < 0:
        h += 360.0

    # 3. Brown rule: warm sector + low chroma + darkish L
    if h < BROWN_HUE_MAX and chroma < BROWN_CHROMA_MAX and L < BROWN_L_MAX:
        return 'brown'

    # 4 & 5. Hue bucketing + pink/red lightness split
    if h >= H_PURPLE_PINK:        # [340°, 360°)
        return 'pink' if L >= PINK_L_MIN else 'red'
    elif h < H_PINK_RED:          # [0°, 16°) — wraps through 0°
        return 'pink' if L >= PINK_L_MIN else 'red'
    elif h < H_RED_ORANGE:        # [16°, 51°)
        return 'red'
    elif h < H_ORANGE_YELLOW:     # [51°, 84°)
        return 'orange'
    elif h < H_YELLOW_GREEN:      # [84°, 119°)
        return 'yellow'
    elif h < H_GREEN_BLUE:        # [119°, 220°) — includes cyan
        return 'green'
    elif h < H_BLUE_PURPLE:       # [220°, 315°)
        return 'blue'
    else:                         # [315°, 340°)
        return 'purple'

# ── Feature vector for SVM ────────────────────────────────────────────────────

def make_feature(L: float, A: float, B: float) -> list:
    """
    5 features matching what color_detector_svm.dart feeds to the ONNX model:
      L      — OpenCV-scale L (0-255)
      A      — OpenCV-scale A (0-255, centred at 128) — raw, NOT shifted
      B      — OpenCV-scale B (0-255, centred at 128) — raw, NOT shifted
      chroma — sqrt((A-128)^2 + (B-128)^2)
      hue    — atan2(B-128, A-128) in radians [-pi, pi]
    """
    a_c    = A - 128.0
    b_c    = B - 128.0
    chroma = math.sqrt(a_c ** 2 + b_c ** 2)
    hue    = math.atan2(b_c, a_c)
    return [L, A, B, chroma, hue]

# ── Dataset loader ────────────────────────────────────────────────────────────

def load_dataset(dataset_dir: Path, csv_file: Path):
    rows = []
    with open(csv_file, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if not rows:
        raise ValueError(f"labels.csv is empty: {csv_file}")

    features   = []
    labels     = []
    rule_preds = []
    failed     = 0

    print(f"Loading {len(rows)} images and extracting features...")
    for i, row in enumerate(rows):
        img_path = dataset_dir / row['image_path']
        label    = row['label']

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            failed += 1
            continue

        try:
            L, A, B = extract_dominant_lab(bgr)
        except Exception:
            failed += 1
            continue

        features.append(make_feature(L, A, B))
        labels.append(label)
        rule_preds.append(classify_rules(L, A, B))

        if (i + 1) % 50 == 0 or (i + 1) == len(rows):
            print(f"  {i+1:>4} / {len(rows)}")

    if failed:
        print(f"  ⚠  {failed} images skipped (could not be read)")

    print(f"\n  {len(features)} samples ready.\n")
    return np.array(features, dtype=np.float32), labels, rule_preds

# ── Confusion matrix plot ─────────────────────────────────────────────────────

def save_confusion_matrix(y_true, y_pred, title: str,
                           out_path: Path, all_labels: list):
    cm   = confusion_matrix(y_true, y_pred, labels=all_labels)
    fig, ax = plt.subplots(figsize=(11, 8))
    disp = ConfusionMatrixDisplay(cm, display_labels=all_labels)
    disp.plot(ax=ax, cmap='Blues', colorbar=True, xticks_rotation=45)
    ax.set_title(title, fontsize=13, pad=14)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out_path.name}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate rule-based and train SVM color classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_and_eval.py --dataset ./dataset
  python train_and_eval.py --dataset ./dataset --output ./results
        """
    )
    ap.add_argument('--dataset', required=True,
                    help='Folder produced by build_dataset.py (contains labels.csv)')
    ap.add_argument('--output',  default='results',
                    help='Folder for model + plots (default: ./results)')
    args = ap.parse_args()

    dataset_dir = Path(args.dataset)
    output_dir  = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_file = dataset_dir / 'labels.csv'
    if not csv_file.exists():
        print(f"[Error] labels.csv not found in {dataset_dir}")
        return

    # ── 1. Load dataset & extract features ────────────────────────────────────
    X, y_true, y_rules = load_dataset(dataset_dir, csv_file)

    unique_labels = sorted(set(y_true),
                           key=lambda l: COLOR_ORDER.index(l)
                           if l in COLOR_ORDER else 99)

    # ── 2. Evaluate rule-based approach ───────────────────────────────────────
    sep = "─" * 52
    print(sep)
    print("  RULE-BASED  (current color_detector.dart)")
    print(sep)

    rules_acc = accuracy_score(y_true, y_rules)
    print(f"  Accuracy: {rules_acc:.1%}\n")
    print(classification_report(y_true, y_rules,
                                labels=unique_labels, zero_division=0))
    save_confusion_matrix(
        y_true, y_rules,
        f"Rule-based  ·  {rules_acc:.1%} accuracy",
        output_dir / "confusion_rules.png",
        unique_labels,
    )

    # ── 3. Train SVM via cross-validation ─────────────────────────────────────
    print(f"\n{sep}")
    print("  SVM CLASSIFIER  (5-fold cross-validation)")
    print(sep)

    le    = LabelEncoder().fit(unique_labels)
    y_enc = le.transform(y_true)

    min_class_count = min(y_true.count(l) for l in unique_labels)
    n_splits        = min(5, min_class_count)

    if n_splits < 2:
        print(f"  ⚠  Some classes have < 2 samples — skipping cross-validation.")
        print(f"     Collect more data (aim for at least 10 per class).")
        return

    if n_splits < 5:
        print(f"  ⚠  Using {n_splits}-fold CV (smallest class has {min_class_count} samples)\n")

    svm = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced')
    cv  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    print(f"  Running {n_splits}-fold cross-validation...")
    y_svm_enc = cross_val_predict(svm, X, y_enc, cv=cv)
    y_svm     = le.inverse_transform(y_svm_enc)

    svm_acc = accuracy_score(y_true, y_svm)
    print(f"  Accuracy: {svm_acc:.1%}\n")
    print(classification_report(y_true, y_svm,
                                labels=unique_labels, zero_division=0))
    save_confusion_matrix(
        y_true, y_svm,
        f"SVM  ·  {svm_acc:.1%} accuracy  ·  {n_splits}-fold CV",
        output_dir / "confusion_svm.png",
        unique_labels,
    )

    # ── 4. Summary ─────────────────────────────────────────────────────────────
    delta = svm_acc - rules_acc
    sign  = "+" if delta >= 0 else ""
    print(f"\n{'═'*52}")
    print(f"  SUMMARY")
    print(f"{'═'*52}")
    print(f"  Rule-based (current) :  {rules_acc:.1%}")
    print(f"  SVM (cross-validated):  {svm_acc:.1%}  ({sign}{delta:.1%})")
    print(f"{'═'*52}\n")

    # ── 5. Save final SVM trained on ALL data ──────────────────────────────────
    print("  Training final model on all data...")
    svm.fit(X, y_enc)

    model_path = output_dir / 'color_svm.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({'model': svm, 'label_encoder': le}, f)

    print(f"  Model saved → {model_path.name}")
    print(f"\n  Results folder: {output_dir.resolve()}")
    print(f"  Files:")
    print(f"    confusion_rules.png  — rule-based confusion matrix")
    print(f"    confusion_svm.png    — SVM confusion matrix")
    print(f"    color_svm.pkl        — trained model for deployment\n")


if __name__ == "__main__":
    main()