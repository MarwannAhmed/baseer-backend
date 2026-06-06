import cv2
import numpy as np

MIN_GAP_FRAC    = 0.008
MIN_BLOCK_FRAC  = 0.01
MERGE_GAP_FRAC  = 0.015
PAD_FRAC        = 0.003
DEBUG           = True


def _to_binary(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.max() > 1:
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        binary = img
    return binary

def _find_gaps(projection, total, min_gap_px, density_thresh):
    gaps, start = [], None
    for i, val in enumerate(projection):
        if val <= density_thresh and start is None:
            start = i
        elif val > density_thresh and start is not None:
            if i - start >= min_gap_px:
                gaps.append((start, i))
            start = None
    if start is not None and total - start >= min_gap_px:
        gaps.append((start, total))
    return gaps

def _gaps_to_spans(gaps, total):
    if not gaps:
        return [(0, total)]
    spans, prev = [], 0
    for gs, ge in gaps:
        if gs > prev:
            spans.append((prev, gs))
        prev = ge
    if prev < total:
        spans.append((prev, total))
    return spans

def _merge_close_spans(spans, merge_gap_px):
    if not spans:
        return spans
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s - merged[-1][1] <= merge_gap_px:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]

def extract(img, is_nat=False):
    binary = _to_binary(img)
    img_h, img_w = binary.shape
    inv = 255 - binary

    min_gap_px = max(2, int(max(img_h, img_w) * MIN_GAP_FRAC))
    merge_gap_px = max(2, int(max(img_h, img_w) * MERGE_GAP_FRAC))
    min_block_px = max(4, int(max(img_h, img_w) * MIN_BLOCK_FRAC))
    pad_px = max(1, int(max(img_h, img_w) * PAD_FRAC))

    h_proj = np.sum(inv, axis=1).astype(np.float32)
    v_proj = np.sum(inv, axis=0).astype(np.float32)

    h_thresh = h_proj.max() * 0.02
    v_thresh = v_proj.max() * 0.02

    row_gaps = _find_gaps(h_proj, img_h, min_gap_px, h_thresh)
    col_gaps = _find_gaps(v_proj, img_w, min_gap_px, v_thresh)

    row_spans = _gaps_to_spans(row_gaps, img_h)
    col_spans = _gaps_to_spans(col_gaps, img_w)

    row_spans = _merge_close_spans(row_spans, merge_gap_px)
    col_spans = _merge_close_spans(col_spans, merge_gap_px)

    row_spans = [(s, e) for s, e in row_spans if e - s >= min_block_px]
    col_spans = [(s, e) for s, e in col_spans if e - s >= min_block_px]

    if not row_spans:
        row_spans = [(0, img_h)]
    if not col_spans:
        col_spans = [(0, img_w)]

    blocks = []
    for ys, ye in row_spans:
        for xs, xe in col_spans:
            region = inv[ys:ye, xs:xe]
            if np.sum(region) < (ye - ys) * (xe - xs) * 0.001 * 255:
                continue
            y1 = max(0,     ys - pad_px)
            x1 = max(0,     xs - pad_px)
            y2 = min(img_h, ye + pad_px)
            x2 = min(img_w, xe + pad_px)
            blocks.append((x1, y1, x2, y2))

    blocks.sort(key=lambda b: (b[0], b[1]))
    if not blocks:
        return [img]
    if DEBUG:
        vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 \
              else img.copy()
        for x1, y1, x2, y2 in blocks:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite("output/extract_blocks.png", vis)

        bar_w = 80
        h_bar = np.ones((img_h, bar_w, 3), dtype=np.uint8) * 255
        h_norm = (h_proj / (h_proj.max() + 1e-5) * (bar_w - 4)).astype(int)
        for r in range(img_h):
            cv2.line(h_bar, (0, r), (h_norm[r], r), (180, 180, 180), 1)
        cv2.line(h_bar, (int(h_thresh / (h_proj.max() + 1e-5) * (bar_w - 4)), 0),
                         (int(h_thresh / (h_proj.max() + 1e-5) * (bar_w - 4)), img_h - 1),
                         (0, 0, 255), 1)
        for gs, ge in row_gaps:
            cv2.rectangle(h_bar, (0, gs), (bar_w - 1, ge), (255, 150, 0), -1)
        for s, e in row_spans:
            cv2.rectangle(h_bar, (bar_w - 4, s), (bar_w - 1, e), (0, 200, 0), -1)
        cv2.imwrite("output/extract_h_proj.png", h_bar)

        bar_h = 80
        v_bar = np.ones((bar_h, img_w, 3), dtype=np.uint8) * 255
        v_norm = (v_proj / (v_proj.max() + 1e-5) * (bar_h - 4)).astype(int)
        for c in range(img_w):
            cv2.line(v_bar, (c, bar_h - 1), (c, bar_h - 1 - v_norm[c]), (180, 180, 180), 1)
        thresh_row = bar_h - 1 - int(v_thresh / (v_proj.max() + 1e-5) * (bar_h - 4))
        cv2.line(v_bar, (0, thresh_row), (img_w - 1, thresh_row), (0, 0, 255), 1)
        for gs, ge in col_gaps:
            cv2.rectangle(v_bar, (gs, 0), (ge, bar_h - 1), (255, 150, 0), -1)
        for s, e in col_spans:
            cv2.rectangle(v_bar, (s, 0), (e, 3), (0, 200, 0), -1)
        cv2.imwrite("output/extract_v_proj.png", v_bar)

        print(f"[0] Extraction done: {len(blocks)} blocks found")

    return [img[y1:y2, x1:x2] for x1, y1, x2, y2 in blocks]