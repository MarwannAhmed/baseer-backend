import os
import cv2

from config import CHARS_DIR, WORDS_DIR, OUTPUT_DIR
from svm_classifier import SVMClassifier
# from cnn_classifier import CNNClassifier
# from rf_classifier  import RFClassifier

MIN_CONFIDENCE = 0.5


def _load_char_crops():
    entries = []
    for fname in sorted(os.listdir(CHARS_DIR)):
        if not fname.endswith(".png"):
            continue
        parts    = fname.replace(".png", "").split("_")
        char_idx = int(parts[0])
        frame_idx = int(parts[1][1:])
        word_idx        = int(parts[2][1:])
        x        = int(parts[3][1:])
        entries.append((char_idx, frame_idx, word_idx, x, os.path.join(CHARS_DIR, fname)))
    return sorted(entries, key=lambda e: (e[1], e[2], e[3]))


def _load_word_line_map():
    mapping = {}
    for fname in sorted(os.listdir(WORDS_DIR)):
        if not fname.endswith(".png"):
            continue
        parts     = fname.replace(".png", "").split("_")
        word_idx  = int(parts[0])
        frame_idx = int(parts[1][1:])
        li        = int(parts[2][1:])
        x         = int(parts[3][1:])
        mapping[(frame_idx, word_idx)] = (li, x, frame_idx)
    return mapping


def recognize():
    if not os.path.exists(CHARS_DIR) or not os.listdir(CHARS_DIR):
        print("No character crops found. Run detect first.")
        return ""

    clf     = SVMClassifier()
    # clf     = CNNClassifier()
    # clf     = RFClassifier()
    entries = _load_char_crops()
    if not entries:
        return ""

    imgs = []
    valid_entries = []
    for e in entries:
        img = cv2.imread(e[4], cv2.IMREAD_GRAYSCALE)
        if img is not None and img.size > 0:
            imgs.append(img)
            valid_entries.append(e)

    if not imgs:
        return ""

    results = clf.predict_batch(imgs)
    word_chars = {}
    for (char_idx, frame_idx, word_idx, x, _), candidates in zip(valid_entries, results):
        for i in range(5):
            if i < len(candidates):
                char, conf = candidates[i]
                print(f"Candidate {i+1}: '{char}' (conf={conf:.2f})")
            else:
                print(f"Candidate {i+1}: None")
        char, conf = candidates[0]
        if conf < MIN_CONFIDENCE:
            char = "?"
        word_chars.setdefault((frame_idx, word_idx), []).append((x, char))

    word_strings = {
        key: "".join(ch for _, ch in sorted(chars))
        for key, chars in word_chars.items()
    }

    word_line_map = _load_word_line_map()

    line_words = {}
    for key, text in word_strings.items():
        if key not in word_line_map:
            continue
        li, x, frame_idx = word_line_map[key]
        line_words.setdefault((frame_idx, li), []).append((x, text))

    text_out = "\n".join(
        " ".join(w for _, w in sorted(line_words[li]))
        for li in sorted(line_words)
    )

    with open(f"{OUTPUT_DIR}/recognized.txt", "w", encoding="utf-8") as f:
        f.write(text_out)

    print(f"[3]  Recognition done")
    return text_out


if __name__ == "__main__":
    recognize()
    print(f"Saved: {OUTPUT_DIR}/recognized.txt")