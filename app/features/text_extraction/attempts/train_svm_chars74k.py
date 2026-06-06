import os

import cv2
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score

from app.features.text_extraction.utils.features import extract_batch
from app.features.text_extraction.config import parameters

CHARS74K_PATH = "data/chars74k/EnglishFnt/"
MAX_PER_CLASS = 2000


def load_chars74k(data_path, max_per_class=1000):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Chars74K path not found: {data_path}")
    
    folders = sorted([d for d in os.listdir(data_path) 
                     if os.path.isdir(os.path.join(data_path, d)) and d.startswith('Sample')])
    
    print(f"Found {len(folders)} class folders")
    
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
            print(f"  Skipping unknown folder: {folder}")
            continue
        
        folder_path = os.path.join(data_path, folder)
        img_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
        
        img_files = img_files[:max_per_class]
        
        for img_file in img_files:
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                images.append(img)
                targets.append(char)
        
        print(f"  Loaded {len(img_files)} images for '{char}' (folder: {folder})")
    
    return images, targets

def train():
    os.makedirs(parameters["train_dirs"]["models"], exist_ok=True)
    
    samples, targets = load_chars74k(CHARS74K_PATH, MAX_PER_CLASS)
    
    unique_classes = sorted(set(targets))
    X = extract_batch(samples)
    
    encoder = LabelEncoder()
    y = encoder.fit_transform(targets)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        probability=True,
        decision_function_shape="ovr",
        class_weight="balanced",
    )
    clf.fit(X_scaled, y)
    
    scores = cross_val_score(clf, X_scaled, y, cv=3, scoring="accuracy", n_jobs=-1)
    print(f"\nCross-val accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    
    with open(parameters["SVM"]["model_path"], "wb") as f:
        pickle.dump(clf, f)
    with open(parameters["SVM"]["scaler_path"], "wb") as f:
        pickle.dump(scaler, f)
    with open(parameters["SVM"]["encoder_path"], "wb") as f:
        pickle.dump(encoder, f)


if __name__ == "__main__":
    train()