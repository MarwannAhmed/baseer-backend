import os

import cv2
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score

from app.features.text_extraction.utils.features import extract_batch
from app.features.text_extraction.config import parameters

# Path to Chars74K dataset
CHARS74K_PATH = "data/chars74k/EnglishFnt/"
MAX_PER_CLASS = 2000


def load_chars74k(data_path, max_per_class=1000):
    """
    Load Chars74K dataset from local folder structure:
    data_path/Sample001/, data_path/Sample002/, etc.
    """    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Chars74K path not found: {data_path}")
    
    # Get all Sample folders
    folders = sorted([d for d in os.listdir(data_path) 
                     if os.path.isdir(os.path.join(data_path, d)) and d.startswith('Sample')])
    
    print(f"Found {len(folders)} class folders")
    
    images = []
    targets = []
    
    for folder in folders:
        # Parse sample number: "Sample011" -> 11
        sample_num = int(folder.replace('Sample', ''))
        
        # Map to character
        if 1 <= sample_num <= 10:
            char = str(sample_num - 1)  # Digits 0-9
        elif 11 <= sample_num <= 36:
            char = chr(ord('A') + (sample_num - 11))  # Uppercase A-Z
        elif 37 <= sample_num <= 62:
            char = chr(ord('a') + (sample_num - 37))  # Lowercase a-z
        else:
            print(f"  Skipping unknown folder: {folder}")
            continue
        
        folder_path = os.path.join(data_path, folder)
        img_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
        
        # Limit samples per class
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
    
    print("=" * 50)
    print("Loading Chars74K dataset...")
    print("=" * 50)
    
    samples, targets = load_chars74k(CHARS74K_PATH, MAX_PER_CLASS)
    
    unique_classes = sorted(set(targets))
    print(f"\n  Total samples: {len(samples)}")
    print(f"  Unique classes: {len(unique_classes)}")
    print(f"  Classes: {unique_classes}")
    
    print("\n" + "=" * 50)
    print("Extracting features...")
    print("=" * 50)
    X = extract_batch(samples)
    
    encoder = LabelEncoder()
    y = encoder.fit_transform(targets)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("\n" + "=" * 50)
    print("Training SVM...")
    print("=" * 50)
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
    
    print(f"\nSaved: {parameters['SVM']['model_path']}")
    print(f"Saved: {parameters['SVM']['scaler_path']}")
    print(f"Saved: {parameters['SVM']['encoder_path']}")


if __name__ == "__main__":
    train()