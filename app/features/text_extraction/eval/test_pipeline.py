import os
import cv2

from app.features.text_extraction.extract.extract import extract
from app.features.text_extraction.preprocess.preprocess import preprocess
from app.features.text_extraction.detect.detect import detect
from app.features.text_extraction.recognize.recognize import recognize

RESULTS_DIR = "data/test/old-books/results"

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for img_idx in range(55):
        image_id = f"c{img_idx + 1:03d}"
        if not os.path.exists(f"data/test/old-books/images/{image_id}.tiff"):
            continue
        img      = cv2.imread(f"data/test/old-books/images/{image_id}.tiff")
        frames   = extract(img)

        all_char_entries  = []
        all_word_line_map = {}

        for frame_idx, frame in enumerate(frames):
            binary = preprocess(frame, frame_number=frame_idx)
            char_entries, word_line_map = detect(binary, frame_number=frame_idx)
            all_char_entries.extend(char_entries)
            all_word_line_map.update(word_line_map)

        res = recognize(char_entries=all_char_entries, word_line_map=all_word_line_map)

        with open(os.path.join(RESULTS_DIR, f"{image_id}.txt"), "w", encoding="utf-8") as f:
            f.write(res)

if __name__ == "__main__":
    main()