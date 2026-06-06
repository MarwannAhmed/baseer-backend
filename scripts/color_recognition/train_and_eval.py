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
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import KMeans

GW_CLAMP_MIN = 0.75
GW_CLAMP_MAX = 1.50
CHROMA_FILTER_THRESHOLD = 10.0
NEUTRAL_CHROMA_THRESHOLD = 15.0
WHITE_L_THRESHOLD = 210.0
BLACK_L_THRESHOLD = 50.0
KMEANS_K = 3
KMEANS_MAX_ITER = 20
PIXEL_STRIDE = 3
BROWN_HUE_MAX = 83.0
BROWN_CHROMA_MAX = 45.0
BROWN_L_MAX = 145.0
PINK_L_MIN = 160.0
H_PINK_RED = 16.0
H_RED_ORANGE = 51.0
H_ORANGE_YELLOW = 84.0
H_YELLOW_GREEN = 119.0
H_GREEN_BLUE = 220.0
H_BLUE_PURPLE = 315.0
H_PURPLE_PINK = 340.0

COLOR_ORDER = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown', 'white', 'gray', 'black']

def gray_world_awb(bgr):
    r = bgr[:, :, 2].astype(np.float64)
    g = bgr[:, :, 1].astype(np.float64)
    b = bgr[:, :, 0].astype(np.float64)

    eps = 1e-6
    r_mean = r.mean(); g_mean = g.mean(); b_mean = b.mean()
    gray_avg = (r_mean + g_mean + b_mean) / 3.0

    r_scale = float(np.clip(gray_avg / (r_mean + eps), GW_CLAMP_MIN, GW_CLAMP_MAX))
    g_scale = float(np.clip(gray_avg / (g_mean + eps), GW_CLAMP_MIN, GW_CLAMP_MAX))
    b_scale = float(np.clip(gray_avg / (b_mean + eps), GW_CLAMP_MIN, GW_CLAMP_MAX))

    out = bgr.astype(np.float64)
    out[:, :, 2] = np.clip(out[:, :, 2] * r_scale, 0, 255)
    out[:, :, 1] = np.clip(out[:, :, 1] * g_scale, 0, 255)
    out[:, :, 0] = np.clip(out[:, :, 0] * b_scale, 0, 255)
    return out.astype(np.uint8)

def extract_dominant_lab(bgr_crop):
    lab = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    pixels = lab[::PIXEL_STRIDE, ::PIXEL_STRIDE].reshape(-1, 3)

    a_c = pixels[:, 1] - 128.0
    b_c = pixels[:, 2] - 128.0
    chroma = np.sqrt(a_c ** 2 + b_c ** 2)
    saturated = pixels[chroma >= CHROMA_FILTER_THRESHOLD]
    if len(saturated) < 10:
        saturated = pixels

    k = min(KMEANS_K, len(saturated))
    if k < 2:
        dominant = saturated.mean(axis=0)
    else:
        km = KMeans(n_clusters=k, init='k-means++', n_init=1, max_iter=KMEANS_MAX_ITER, random_state=0)
        km.fit(saturated)
        counts = np.bincount(km.labels_)
        dominant = km.cluster_centers_[counts.argmax()]

    return float(dominant[0]), float(dominant[1]), float(dominant[2])

def classify_rules(L: float, rawA: float, rawB: float) -> str:
    A = rawA - 128.0
    B = rawB - 128.0
    chroma = math.sqrt(A * A + B * B)
    
    if chroma < NEUTRAL_CHROMA_THRESHOLD:
        if L > WHITE_L_THRESHOLD: return 'white'
        if L < BLACK_L_THRESHOLD: return 'black'
        return 'gray'
    h = math.degrees(math.atan2(B, A))
    if h < 0:
        h += 360.0

    if h < BROWN_HUE_MAX and chroma < BROWN_CHROMA_MAX and L < BROWN_L_MAX:
        return 'brown'
    if h >= H_PURPLE_PINK:
        return 'pink' if L >= PINK_L_MIN else 'red'
    elif h < H_PINK_RED:
        return 'pink' if L >= PINK_L_MIN else 'red'
    elif h < H_RED_ORANGE:
        return 'red'
    elif h < H_ORANGE_YELLOW:
        return 'orange'
    elif h < H_YELLOW_GREEN:  
        return 'yellow'
    elif h < H_GREEN_BLUE:
        return 'green'
    elif h < H_BLUE_PURPLE:    
        return 'blue'
    else:                      
        return 'purple'

def make_feature(L, A, B):
    a_c = A - 128.0
    b_c = B - 128.0
    chroma = math.sqrt(a_c ** 2 + b_c ** 2)
    hue = math.atan2(b_c, a_c)
    return [L, A, B, chroma, hue]

def load_dataset(dataset_dir: Path, csv_file: Path):
    rows = []
    with open(csv_file, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if not rows:
        raise ValueError(f"labels.csv is empty: {csv_file}")

    features = []
    labels = []
    rule_preds = []
    failed = 0

    for i, row in enumerate(rows):
        img_path = dataset_dir / row['image_path']
        label = row['label']
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
        print(f"{failed} images failed and skipped")

    print(f"\n  {len(features)} samples ready.\n")
    return np.array(features, dtype=np.float32), labels, rule_preds

def save_confusion_matrix(y_true, y_pred, title, out_path, all_labels):
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    fig, ax = plt.subplots(figsize=(11, 8))
    disp = ConfusionMatrixDisplay(cm, display_labels=all_labels)
    disp.plot(ax=ax, cmap='Blues', colorbar=True, xticks_rotation=45)
    ax.set_title(title, fontsize=13, pad=14)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)

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
    X, y_true, y_rules = load_dataset(dataset_dir, csv_file)

    unique_labels = sorted(set(y_true), key=lambda l: COLOR_ORDER.index(l) if l in COLOR_ORDER else 99)

    rules_acc = accuracy_score(y_true, y_rules)
    print(f"  Accuracy: {rules_acc:.1%}\n")
    print(classification_report(y_true, y_rules, labels=unique_labels, zero_division=0))
    save_confusion_matrix(
        y_true, y_rules,
        f"Rule-based  {rules_acc:.1%} accuracy",
        output_dir / "confusion_rules.png",
        unique_labels,
    )
    print("SVM CLASSIFIER:")

    le    = LabelEncoder().fit(unique_labels)
    y_enc = le.transform(y_true)

    min_class_count = min(y_true.count(l) for l in unique_labels)
    n_splits        = min(5, min_class_count)

    if n_splits < 2:
        return

    svm = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced')
    cv  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    print(f"  Running {n_splits}-fold cross-validation")
    y_svm_enc = cross_val_predict(svm, X, y_enc, cv=cv)
    y_svm     = le.inverse_transform(y_svm_enc)

    svm_acc = accuracy_score(y_true, y_svm)
    print(f"Accuracy: {svm_acc:.1%}\n")
    print(classification_report(y_true, y_svm,
                                labels=unique_labels, zero_division=0))
    save_confusion_matrix(
        y_true, y_svm,
        f"SVM {svm_acc:.1%} accuracy {n_splits}-fold CV",
        output_dir / "confusion_svm.png",
        unique_labels,
    )
    delta = svm_acc - rules_acc
    sign  = "+" if delta >= 0 else ""
    print(f"  Rule-based (current) :  {rules_acc:.1%}")
    print(f"  SVM (cross-validated):  {svm_acc:.1%}  ({sign}{delta:.1%})")
    print("  Training final model on all data")
    svm.fit(X, y_enc)

    model_path = output_dir / 'color_svm.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({'model': svm, 'label_encoder': le}, f)

if __name__ == "__main__":
    main()