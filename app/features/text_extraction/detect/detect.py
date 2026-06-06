import cv2
import numpy as np

from app.features.text_extraction.config import output_paths
from app.features.text_extraction.detect.lines  import detect_lines
from app.features.text_extraction.detect.words  import detect_words
from app.features.text_extraction.detect.chars  import detect_chars


def _is_valid_word(w, h, area, median_h):
    return (
        area > 100
        and h > 8
        and 0.1  < w    / (h      + 1e-5) < 20.0
        and 0.4  < h    / (median_h + 1e-5) < 1.6
        and 0.08 < area / (w * h  + 1e-5)
    )

def detect(binary, save_output=False, frame_number=0):
    img_h, img_w = binary.shape
    vis          = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    word_idx = char_idx = 0
    char_entries  = []
    word_line_map = {}

    heights = []
    lines, ah = detect_lines(binary)

    for li, (ls, le) in enumerate(lines):
        words = detect_words(binary, ls, le, ah)
        if not words:
            continue

        heights  = [ay2 - ay1 for _, ay1, _, ay2 in words]
        median_h = float(np.median(heights)) if heights else ah

        cv2.rectangle(vis, (0, ls), (img_w - 1, le), (255, 100, 0), 2)

        for ax1, ay1, ax2, ay2 in words:
            w = ax2 - ax1
            h = ay2 - ay1
            area = w * h

            if not _is_valid_word(w, h, area, median_h):
                continue

            cv2.rectangle(vis, (ax1, ay1), (ax2, ay2), (0, 200, 0), 3)

            word_crop = binary[ay1:ay2, ax1:ax2]
            word_line_map[(frame_number, word_idx)] = (li, ax1, frame_number)

            if save_output:
                wfname    = f"{output_paths['words']}/f{frame_number}_l{li}_w{word_idx}_x{ax1}.png"
                cv2.imwrite(wfname, word_crop)

            chars = detect_chars(word_crop, ax1, ay1, ah)
            for cx1, cy1, cx2, cy2 in chars:
                heights.append(cy2 - cy1)
                cv2.rectangle(vis, (cx1, cy1), (cx2, cy2), (0, 0, 200), 2)
                char_crop = binary[cy1:cy2, cx1:cx2]
                padding = h // 4
                h, w = char_crop.shape
                padded_char = np.ones((h + 2*padding, w + 2*padding), dtype=np.uint8) * 255
                padded_char[padding:padding+h, padding:padding+w] = char_crop
                char_entries.append((char_idx, frame_number, word_idx, cx1, padded_char))
                if save_output:
                    cfname    = f"{output_paths['chars']}/f{frame_number}_w{word_idx}_c{char_idx}_x{cx1}.png"
                    cv2.imwrite(cfname, padded_char)
                char_idx += 1

            word_idx += 1

    if save_output:
        cv2.imwrite(f"{output_paths['detect']}/box_{frame_number}.png", vis)
        
    return char_entries, word_line_map