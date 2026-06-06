import os
import cv2
import numpy as np
import joblib
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder, StandardScaler

from features import extract_batch
from config import MODELS_DIR, MODEL_PATH, SCALER_PATH, ENCODER_PATH

CHARS74K_PATH = "data/chars74k/EnglishFnt/"
SYNTHETIC_PATH = "data/synthetic_chars/"
MAX_PER_CLASS = 500

def preprocess_character(img, apply_augmentation=False):
    h, w = img.shape
    
    scale = 32 / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    pad_h = (32 - new_h) // 2
    pad_w = (32 - new_w) // 2
    img = cv2.copyMakeBorder(img, pad_h, 32 - new_h - pad_h, pad_w, 32 - new_w - pad_w, cv2.BORDER_CONSTANT, value=255)
    
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    
    if apply_augmentation:
        if np.random.random() > 0.5:
            noise = np.random.random(img.shape)
            img[noise < 0.02] = 0
            img[noise > 0.98] = 255
        
        if np.random.random() > 0.7:
            img = cv2.GaussianBlur(img, (3, 3), 0.5)
            _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
        if np.random.random() > 0.6:
            M = cv2.getRotationMatrix2D((16, 16), np.random.uniform(-3, 3), 1)
            img = cv2.warpAffine(img, M, (32, 32), borderValue=255)
        
        if np.random.random() > 0.5:
            shift_x = np.random.randint(-1, 2)
            shift_y = np.random.randint(-1, 2)
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            img = cv2.warpAffine(img, M, (32, 32), borderValue=255)
    
    return img

def load_chars74k(data_path, max_per_class=500):
    if not os.path.exists(data_path):
        print(f"Chars74K path not found: {data_path}")
        return [], []
    
    folders = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d)) and d.startswith('Sample')])
    
    print(f"Found {len(folders)} Chars74K class folders")
    
    images = []
    targets = []
    
    for folder in folders:
        sample_num = int(folder.replace('Sample', ''))
        
        if 1 <= sample_num <= 10:
            char = str(sample_num - 1)
        elif 11 <= sample_num <= 36:
            char = chr(ord('A') + (sample_num - 11))
        elif 37 <= sample_num <= 62:
            char = chr(ord('a') + (sample_num - 37))
        else:
            continue
        
        folder_path = os.path.join(data_path, folder)
        img_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
        img_files = img_files[:max_per_class]
        
        for img_file in img_files:
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                processed = preprocess_character(img, apply_augmentation=False)
                images.append(processed)
                targets.append(char)
        
        print(f"  Loaded {len(img_files)} Chars74K images for '{char}'")
    
    return images, targets

def load_synthetic_dataset(data_path, max_per_class=500, augment_ratio=0.3):
    if not os.path.exists(data_path):
        print(f"Synthetic dataset not found at: {data_path}")
        return [], []
    
    folders = sorted([d for d in os.listdir(data_path) 
                     if os.path.isdir(os.path.join(data_path, d)) and d.startswith('Sample')])
    
    print(f"Found {len(folders)} synthetic class folders")
    
    images = []
    targets = []
    
    for folder in folders:
        sample_num = int(folder.replace('Sample', ''))
        
        if 1 <= sample_num <= 10:
            char = str(sample_num - 1)
        elif 11 <= sample_num <= 36:
            char = chr(ord('A') + (sample_num - 11))
        elif 37 <= sample_num <= 62:
            char = chr(ord('a') + (sample_num - 37))
        else:
            continue
        
        folder_path = os.path.join(data_path, folder)
        img_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
        img_files = img_files[:max_per_class]
        
        for img_file in img_files:
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                processed = preprocess_character(img, apply_augmentation=False)
                images.append(processed)
                targets.append(char)
                
                if np.random.random() < augment_ratio:
                    augmented = preprocess_character(img, apply_augmentation=True)
                    images.append(augmented)
                    targets.append(char)
        
        print(f"  Loaded {len(img_files)} synthetic images for '{char}'")
    
    return images, targets

def train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print("Loading Chars74K dataset")
    chars74k_images, chars74k_targets = load_chars74k(CHARS74K_PATH, MAX_PER_CLASS)
    
    print("Loading synthetic dataset")
    synthetic_images, synthetic_targets = load_synthetic_dataset(SYNTHETIC_PATH, MAX_PER_CLASS, augment_ratio=0.3)
    
    all_images = chars74k_images + synthetic_images
    all_targets = chars74k_targets + synthetic_targets
    
    unique_classes = sorted(set(all_targets))
    
    print("Extracting features")
    X = extract_batch(all_images)
    
    encoder = LabelEncoder()
    y = encoder.fit_transform(all_targets)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Training SVM")
    _base = LinearSVC(
        C=1.0,
        max_iter=2000,
        class_weight="balanced",
        verbose=True,
    )
    clf = CalibratedClassifierCV(_base, cv=3)
    clf.fit(X_scaled, y)
    
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(clf, X_scaled, y, cv=3, scoring="accuracy", n_jobs=-1)
    print(f"\nCross-val accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    joblib.dump(clf,     MODEL_PATH,   compress=3)
    joblib.dump(scaler,  SCALER_PATH,  compress=3)
    joblib.dump(encoder, ENCODER_PATH, compress=3)
    
    print(f"\nSaved: {MODEL_PATH}")
    print(f"Saved: {SCALER_PATH}")
    print(f"Saved: {ENCODER_PATH}")

if __name__ == "__main__":
    train()