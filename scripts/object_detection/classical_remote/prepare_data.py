import os
import json
import random

import numpy as np
import cv2
from tqdm import tqdm

from app.features.object_detection.features import extract_features
from app.features.object_detection.utils import compute_iou

random.seed(42)
np.random.seed(42)


def load_coco(annotation_file):
    with open(annotation_file, "r") as file:
        data = json.load(file)

    categories = {category["id"]: category["name"] for category in data["categories"]}
    images = {image["id"]: image for image in data["images"]}
    annotations = data["annotations"]

    return categories, images, annotations


IMAGE_INDEX: dict = {}


def build_image_index(images_dir):
    global IMAGE_INDEX
    if IMAGE_INDEX:
        return

    count = 0
    for root, _, files in os.walk(images_dir):
        for file_name in files:
            if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                IMAGE_INDEX[file_name] = os.path.join(root, file_name)
                count += 1


def find_image_path(image_info, images_dir):

    build_image_index(images_dir)

    file_name = image_info["file_name"]
    base_name = os.path.basename(file_name)

    if base_name in IMAGE_INDEX:
        return IMAGE_INDEX[base_name]

    for candidate in [
        os.path.join(images_dir, file_name),
        os.path.join(images_dir, "train2017", base_name),
        os.path.join(images_dir, "val2017", base_name),
    ]:
        if os.path.isfile(candidate):
            return candidate

    return None


def extract_positives(categories, images, annotations, images_dir):
    class_names = ["person", "bottle", "chair", "dining table", "handbag"]
    name_to_id = {
        name: category_id
        for category_id, name in categories.items()
        if name in class_names
    }

    by_class = {class_name: [] for class_name in class_names}
    for annotation in annotations:
        name = categories.get(annotation["category_id"])
        if name in by_class:
            by_class[name].append(annotation)

    positives = {}

    for class_name in class_names:
        annotations = by_class[class_name]
        random.shuffle(annotations)

        features = []
        for annotation in tqdm(annotations, desc=f"cropping", leave=False):
            if len(features) >= 5000:
                break

            x, y, width, height = annotation["bbox"]
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + width), int(y + height)

            if width < 32 or height < 32:
                continue

            info = images.get(annotation["image_id"])
            if info is None:
                continue

            image_path = find_image_path(info, images_dir)
            if image_path is None:
                continue

            image = cv2.imread(image_path)
            if image is None:
                continue

            image_height, image_width = image.shape[:2]

            pad = 16
            padded_left = max(0, x1 - pad)
            padded_top = max(0, y1 - pad)
            padded_right = min(image_width, x2 + pad)
            padded_bottom = min(image_height, y2 + pad)

            patch = image[padded_top:padded_bottom, padded_left:padded_right]
            if patch.size == 0:
                continue

            patches_to_add = [patch]
            patches_to_add.append(cv2.flip(patch, 1))
            jitter_range = np.random.randint(-30, 31)
            brightness_jitter = np.clip(
                patch.astype(np.int16) + jitter_range, 0, 255
            ).astype(np.uint8)
            patches_to_add.append(brightness_jitter)

            for patch in patches_to_add:
                if len(features) >= 5000:
                    break
                try:
                    feature = extract_features(patch)
                    features.append(feature)
                except Exception as error:
                    print(error)
                    continue

        positives[class_name] = np.array(features, dtype=np.float32)

    return positives


def extract_negatives(images, annotations, images_dir):
    image_boxes = {}
    for annotation in annotations:
        image_id = annotation["image_id"]
        x, y, box_width, box_height = annotation["bbox"]
        image_boxes.setdefault(image_id, []).append(
            [int(x), int(y), int(x + box_width), int(y + box_height)]
        )

    all_ids = list(images.keys())
    random.shuffle(all_ids)

    window_width, window_height = (96, 96)
    features = []

    for id in tqdm(all_ids, desc="sampling", leave=False):
        if len(features) >= 5000:
            break

        info = images[id]
        image_path = find_image_path(info, images_dir)
        if image_path is None:
            continue

        image = cv2.imread(image_path)
        if image is None:
            continue

        image_height, image_width = image.shape[:2]
        if image_width < window_width or image_height < window_height:
            continue

        existing_boxes = image_boxes.get(id, [])

        for _ in range(4):
            if len(features) >= 5000:
                break

            x1 = random.randint(0, image_width - window_width)
            y1 = random.randint(0, image_height - window_height)
            x2 = x1 + window_width
            y2 = y1 + window_height

            overlaps = any(
                compute_iou([x1, y1, x2, y2], box) > 0.3 for box in existing_boxes
            )
            if overlaps:
                continue

            patch = image[y1:y2, x1:x2]
            try:
                feature_vector = extract_features(patch)
                features.append(feature_vector)
            except Exception as error:
                print(error)
                continue

    return np.array(features, dtype=np.float32)


def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    annotations_path = os.path.join("coco", "annotations")
    annotation_file = os.path.join(annotations_path, "instances_train2017.json")

    if not os.path.isfile(annotation_file):
        print(f"Annotation file not found at {annotation_file}")
        return

    images_path = os.path.join("coco", "images")
    if not os.path.isdir(images_path):
        print(f"Images directory not found at {images_path}")
        return

    categories, images, annotations = load_coco(annotation_file)

    positives = extract_positives(categories, images, annotations, images_path)
    negatives = extract_negatives(images, annotations, images_path)

    class_names = ["person", "bottle", "chair", "dining table", "handbag"]
    for class_name in class_names:
        positive_array = positives[class_name]
        path = os.path.join("data", f"positives_{class_name}.npy")
        np.save(path, positive_array)

    negative_path = os.path.join("data", "negatives.npy")
    np.save(negative_path, negatives)

    image_paths = []
    for image_id, info in images.items():
        path = find_image_path(info, images_path)
        if path:
            image_paths.append(path)
        if len(image_paths) >= 600:
            break

    list_path = os.path.join("data", "image_list.json")
    with open(list_path, "w") as f:
        json.dump(image_paths, f)


if __name__ == "__main__":
    main()
