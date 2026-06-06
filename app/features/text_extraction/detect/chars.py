import cv2
import numpy as np

from app.features.text_extraction.config import parameters


def _cut_candidates(blob_inv, midpoint_radius=4):
    fg = (blob_inv > 0).astype(np.uint8)
    dist = cv2.distanceTransform(fg, cv2.DIST_L2, 3)
    col_thickness = dist.max(axis=0)
    w = col_thickness.shape[0]

    valleys = []
    if col_thickness.max() > 0:
        norm    = col_thickness / col_thickness.max()
        valleys = [x for x in range(1, w - 1)
                   if norm[x] < parameters["chars"]["chop_min_valley"]
                   and norm[x] <= norm[x - 1]
                   and norm[x] <= norm[x + 1]]

    mid= w // 2
    q1 = w // 4
    q3 = (3 * w) // 4
    
    mid_window = [x for x in range(mid - midpoint_radius, mid + midpoint_radius + 1) if 1 <= x <= w - 1]
    q1_window = [x for x in range(q1 - midpoint_radius, q1 + midpoint_radius + 1) if 1 <= x <= w - 1]
    q3_window = [x for x in range(q3 - midpoint_radius, q3 + midpoint_radius + 1) if 1 <= x <= w - 1]

    return sorted(set(valleys + mid_window + q1_window + q3_window))

def _best_cut(cut_xs, start, w, min_cw, ah, relaxed=False):
    best_end, best_score = None, float("inf")
    for end in (x for x in cut_xs if start < x <= w):
        seg_w     = end - start
        remaining = w - end
        if not relaxed:
            if seg_w < min_cw:
                continue
            if 0 < remaining < min_cw:
                continue
        score = abs(seg_w - ah)
        if score < best_score:
            best_score = score
            best_end   = end
    return best_end

def _chop_blob(blob_inv, blob_x, blob_y, ah):
    h, w   = blob_inv.shape
    min_cw = int(parameters["chars"]["min_char_width"] * ah * 5)

    if w <= parameters["chars"]["max_char_width"] * ah:
        return [(blob_x, blob_y, blob_x + w, blob_y + h)]

    cuts = _cut_candidates(blob_inv)
    if not cuts:
        return [(blob_x, blob_y, blob_x + w, blob_y + h)]

    cut_xs= [0] + cuts + [w]
    result = []
    start = 0

    while start < w:
        best_end = (_best_cut(cut_xs, start, w, min_cw, ah, relaxed=False)
                    or _best_cut(cut_xs, start, w, min_cw, ah, relaxed=True)
                    or w)

        seg_w = best_end - start
        if seg_w < w:
            sub_segments = _chop_blob(
                blob_inv[:, start:best_end],
                blob_x + start,
                blob_y,
                ah,
            )
        else:
            sub_segments = [(blob_x + start, blob_y, blob_x + best_end, blob_y + h)]

        result.extend(sub_segments)
        start = best_end

    return result

def _valid_char(x1, y1, x2, y2, ah):
    w, h = x2 - x1, y2 - y1
    if h == 0 or w == 0:
        return False
    return (
        parameters["average"]["height_min"] * ah < h < parameters["average"]["height_max"] * ah
        and w * h > parameters["average"]["area_min"] * ah * ah
        and parameters["chars"]["min_char_width"] * ah < w < parameters["chars"]["max_char_width"] * ah
    )

def detect_chars(word_binary, word_x, word_y, ah):
    inv    = 255 - word_binary
    n, _, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    max_single_w   = int(parameters["chars"]["max_char_width"] * ah)

    all_comps, chars = [], []

    for i in range(1, n):
        bx, by, bw, bh, area = stats[i]
        ax1, ay1 = word_x + bx, word_y + by
        ax2, ay2 = ax1 + bw, ay1 + bh
        all_comps.append((ax1, ay1, ax2, ay2))
        if bh < parameters["average"]["height_min"] * ah or area < parameters["average"]["area_min"] * ah * ah:
            continue
        if bw <= max_single_w:
            chars.append((ax1, ay1, ax2, ay2))
        else:
            chars.extend(_chop_blob(inv[by:by+bh, bx:bx+bw], ax1, ay1, ah))

    chars.sort(key=lambda c: c[0])
    valid = [c for c in chars if _valid_char(*c, ah)]

    absorbed, merged = set(), []
    for char in valid:
        box = char
        for comp in all_comps:
            if comp in absorbed or comp == char:
                continue
            cx1, cy1, cx2, cy2 = comp
            if cx2 > box[0] and cx1 < box[2] and cy2 <= box[1]:
                absorbed.add(comp)
                box = (min(box[0], cx1), cy1, max(box[2], cx2), box[3])
        merged.append((char, box))

    return [box for orig, box in merged if orig not in absorbed]