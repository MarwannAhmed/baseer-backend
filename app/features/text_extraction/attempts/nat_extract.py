import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

WINDOW_FRAC      = 0.1
MIN_EDGES        = 6
MAX_CV           = 0.70
MIN_MEAN_DIST    = 1.0
EDGE_THRESHOLD   = 125
VERTICAL_BLUR    = 0.04
FULL_COVERAGE    = 0.70
EXPAND_FRAC      = 0.04
MIN_CLUSTER_FRAC = 0.005
MERGE_GAP_FRAC   = 0.05
ROW_BOOST_FRAC   = 0.15
DEBUG            = True


def _to_gray(img):
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def _edge_map(gray):
    sobel = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    edges = (np.abs(sobel) > EDGE_THRESHOLD).astype(np.uint8) * 255
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=1)
    if DEBUG:
        cv2.imwrite("output/extract_edges.png", edges)
    return (edges > 0).astype(np.uint8)

def _score_window(local_edges):
    if len(local_edges) < MIN_EDGES:
        return 0.0
    gaps   = np.diff(local_edges).astype(np.float32)
    median = np.median(gaps)
    if median <= 0:
        return 0.0
    intra = gaps[gaps <= median]
    inter = gaps[gaps >  median]
    if len(intra) < 1 or len(inter) < 1:
        return 0.0
    mean_intra = float(np.mean(intra))
    mean_inter = float(np.mean(inter))
    if mean_intra < MIN_MEAN_DIST or mean_inter < MIN_MEAN_DIST:
        return 0.0
    if mean_inter <= mean_intra:
        return 0.0
    separation_ratio = mean_inter / (mean_intra + 1e-5)
    separation_score = 1.0 - 1.0 / separation_ratio
    cv_inter = float(np.std(inter)) / (mean_inter + 1e-5)
    if cv_inter > MAX_CV:
        return 0.0
    regularity_score = 1.0 - cv_inter / MAX_CV
    return separation_score * 0.5 + regularity_score * 0.5

def _score_row(edge_row, window_w, max_mean_dist):
    n = len(edge_row)
    scores = np.zeros(n, dtype=np.float32)
    epos = np.where(edge_row)[0]
    max_edges = max(MIN_EDGES + 1, window_w // 3)
    if len(epos) < MIN_EDGES:
        return scores
    step = max(1, window_w // 4)
    for cx in range(window_w // 2, n - window_w // 2, step):
        mask        = (epos >= cx - window_w // 2) & (epos < cx + window_w // 2)
        local_edges = epos[mask]
        ne          = len(local_edges)
        if ne < MIN_EDGES or ne > max_edges:
            continue
        if float(np.mean(np.diff(local_edges))) > max_mean_dist:
            continue
        score = _score_window(local_edges)
        if score > 0:
            x1 = max(0, cx - window_w // 2)
            x2 = min(n, cx + window_w // 2)
            scores[x1:x2] = np.maximum(scores[x1:x2], score)
    return scores

def _build_score_map(edges, img_h, img_w):
    window_w = max(40, int(img_w * WINDOW_FRAC))
    max_mean_dist = img_w * 0.08
    score_map = np.zeros((img_h, img_w), dtype=np.float32)

    for r in range(img_h):
        score_map[r] = _score_row(edges[r], window_w, max_mean_dist)

    if DEBUG:
        raw_viz = (score_map / (score_map.max() + 1e-5) * 255).astype(np.uint8)
        raw_viz_color = cv2.applyColorMap(raw_viz, cv2.COLORMAP_JET)
        cv2.imwrite("output/extract_score_raw.png", raw_viz_color)

    sigma = max(3, int(img_h * VERTICAL_BLUR))
    score_map = gaussian_filter1d(score_map, sigma=sigma, axis=0)
    if score_map.max() > 0:
        score_map /= score_map.max()

    if DEBUG:
        score_viz       = (score_map * 255).astype(np.uint8)
        score_viz_color = cv2.applyColorMap(score_viz, cv2.COLORMAP_JET)
        cv2.imwrite("output/extract_score_map.png", score_viz_color)

    return score_map

def _otsu_threshold(score_map):
    score_u8 = (score_map * 255).astype(np.uint8)
    otsu_val, _ = cv2.threshold(score_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(otsu_val) / 255.0

def _find_clusters(score_map, img_h, img_w):
    threshold = _otsu_threshold(score_map)
    keep = (score_map > threshold).astype(np.uint8) * 255

    if DEBUG:
        cv2.imwrite("output/extract_keep_mask.png", keep)

    n, _, stats, _ = cv2.connectedComponentsWithStats(keep, connectivity=8)
    min_area = img_h * img_w * MIN_CLUSTER_FRAC
    merge_gap = max(img_h, img_w) * MERGE_GAP_FRAC
    boxes = []

    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        boxes.append([x, y, x + w, y + h])

    if not boxes:
        return []

    boxes.sort(key=lambda b: (b[1], b[0]))
    merged = [boxes[0]]
    for bx1, by1, bx2, by2 in boxes[1:]:
        mx1, my1, mx2, my2 = merged[-1]
        gap_x = max(0, bx1 - mx2, mx1 - bx2)
        gap_y = max(0, by1 - my2, my1 - by2)
        if gap_x <= merge_gap and gap_y <= merge_gap:
            merged[-1] = [min(mx1, bx1), min(my1, by1),
                          max(mx2, bx2), max(my2, by2)]
        else:
            merged.append([bx1, by1, bx2, by2])

    if DEBUG:
        viz          = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        viz[:, :, 0] = (score_map * 255).astype(np.uint8)
        for b in merged:
            cv2.rectangle(viz, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        cv2.imwrite("output/extract_clusters.png", viz)

    return merged

def nat_extract(img):    
    gray = _to_gray(img)
    img_h, img_w = gray.shape

    edges = _edge_map(gray)
    score_map = _build_score_map(edges, img_h, img_w)
    clusters = _find_clusters(score_map, img_h, img_w)

    if not clusters:
        if DEBUG:
            print("No clusters found, returning full image")
        return [img]

    pad_x = max(1, int(img_w * EXPAND_FRAC // 2))
    pad_y = max(1, int(img_h * EXPAND_FRAC))

    frames, boxes = [], []
    for bx1, by1, bx2, by2 in clusters:
        coverage = ((bx2 - bx1) * (by2 - by1)) / (img_w * img_h)
        if coverage >= FULL_COVERAGE:
            if DEBUG:
                print(f"Cluster covers {coverage:.2f} of image, returning full image")
            return [img]
        x1 = max(0,     bx1 - pad_x)
        y1 = max(0,     by1 - pad_y)
        x2 = min(img_w, bx2 + pad_x)
        y2 = min(img_h, by2 + pad_y)
        frames.append(img[y1:y2, x1:x2])
        boxes.append((x1, y1, x2, y2))

    if DEBUG:
        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.imwrite("output/extract_overlay.png", overlay)

    print(f"[0] Text extraction done, {len(boxes)} regions found")
    return frames