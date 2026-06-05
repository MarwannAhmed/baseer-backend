import os
import cv2

from app.features.text_extraction.dl.inference import infer

RESULTS_DIR = "data/test/synthetic/dl-results"

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for img_idx in range(55):
        image_id = f"page_{img_idx + 1:03d}"
        if not os.path.exists(f"data/test/synthetic/images/{image_id}.png"):
            continue
        img      = cv2.imread(f"data/test/synthetic/images/{image_id}.png")
        res = infer(img)

        with open(os.path.join(RESULTS_DIR, f"{image_id}.txt"), "w", encoding="utf-8") as f:
            f.write(res)

if __name__ == "__main__":
    main()