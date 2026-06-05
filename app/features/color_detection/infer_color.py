import cv2
import math
import pickle
import argparse
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans

PIXEL_STRIDE = 3
CHROMA_FILTER_THRESHOLD = 10.0
KMEANS_K = 3
KMEANS_MAX_ITER = 20

def load_model(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    model = data['model']
    le    = data['label_encoder']
    print(f"[OK] Model loaded from {pkl_path}")
    print(f"     Classes: {list(le.classes_)}\n")
    return model, le

def extract_features(bgr):
    lab    = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    pixels = lab[::PIXEL_STRIDE, ::PIXEL_STRIDE].reshape(-1, 3)

    a_c    = pixels[:, 1] - 128.0
    b_c    = pixels[:, 2] - 128.0
    chroma = np.sqrt(a_c**2 + b_c**2)

    sat = pixels[chroma >= CHROMA_FILTER_THRESHOLD]
    if len(sat) < 10:
        print("Warning: fewer than 10 chromatic pixels, using all pixels")
        sat = pixels

    k  = min(KMEANS_K, len(sat))
    km = KMeans(n_clusters=k, n_init=1, max_iter=KMEANS_MAX_ITER, random_state=0).fit(sat)
    dominant = km.cluster_centers_[np.bincount(km.labels_).argmax()]

    L, A, B = float(dominant[0]), float(dominant[1]), float(dominant[2])
    a_c     = A - 128.0
    b_c     = B - 128.0
    chroma  = math.sqrt(a_c**2 + b_c**2)
    hue     = math.atan2(b_c, a_c)

    features = np.array([[L, A, B, chroma, hue]], dtype=np.float32)
    return features, L, A, B, chroma, hue


def predict(image_path: str, model, le, verbose=True) -> str:
    bgr = cv2.imread(image_path)
    if bgr is None:
        print(f"[ERROR] Could not read {image_path}")
        return "error"

    features, L, A, B, chroma, hue = extract_features(bgr)

    if verbose:
        print(f"Image: {image_path}")
        print(f"L={L:.1f}  A={A:.1f}  B={B:.1f}")
        print(f"chroma={chroma:.1f}  hue={math.degrees(hue):.1f}°")

    label = le.inverse_transform(model.predict(features))[0]

    # Also show top-3 probabilities if model supports it
    if hasattr(model, 'predict_proba'):
        probs     = model.predict_proba(features)[0]
        top3_idx  = probs.argsort()[::-1][:3]
        top3      = [(le.classes_[i], probs[i]) for i in top3_idx]
        prob_str  = "  ".join(f"{n}:{p:.0%}" for n, p in top3)
        if verbose:
            print(f"Top-3: {prob_str}")

    if verbose:
        print(f"→ PREDICTION: {label}\n")

    return label


def main():
    ap = argparse.ArgumentParser(description="Infer color from images using color_svm.pkl")
    ap.add_argument('--model', default='results/color_svm.pkl', help='Path to color_svm.pkl')
    ap.add_argument('images', nargs='*', help='Image file(s) to classify')
    ap.add_argument('--folder', help='Folder of images to classify')
    args = ap.parse_args()

    model, le = load_model(args.model)

    paths = list(args.images or [])
    if args.folder:
        folder = Path(args.folder)
        paths += [str(p) for p in sorted(folder.iterdir())
                  if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')]

    if not paths:
        print("No images provided. Use: python infer_color.py image.jpg")
        print("Or: python infer_color.py --folder ./test_images")
        return

    for p in paths:
        predict(p, model, le, verbose=True)

if __name__ == '__main__':
    main()