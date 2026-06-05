import cv2
import numpy as np

AH_HEIGHT_MIN     = 0.3
AH_HEIGHT_MAX     = 2.5
AH_AREA_MIN       = 0.05
LINE_ASSIGN_FRAC  = 0.6
LINE_PAD_FRAC     = 0.25
MIN_LINE_BLOBS    = 1


def _estimate_ah(binary):
    inv            = 255 - binary
    n, _, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    heights        = [stats[i][3] for i in range(1, n) if stats[i][4] > 5]
    return float(np.median(heights)) if heights else 20.0


def _extract_blobs(binary, ah):
    inv            = 255 - binary
    n, _, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    blobs          = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if (AH_HEIGHT_MIN * ah < h < AH_HEIGHT_MAX * ah
                and area > AH_AREA_MIN * ah * ah):
            blobs.append({
                "x1": x, "y1": y,
                "x2": x + w, "y2": y + h,
                "cx": x + w / 2.0,
                "cy": y + h / 2.0,
                "baseline": y + h,
            })
    return sorted(blobs, key=lambda b: b["cx"])


def _assign_blobs_to_lines(blobs, ah):
    lines = []

    for blob in blobs:
        assigned = False
        for line in lines:
            line_baseline = np.median([b["baseline"] for b in line])
            line_top      = min(b["y1"] for b in line)
            overlap_lo    = max(blob["y1"], line_top)
            overlap_hi    = min(blob["y2"], line_baseline + ah * 0.3)
            overlap       = max(0, overlap_hi - overlap_lo)
            if overlap >= blob["y2"] - blob["y1"] * LINE_ASSIGN_FRAC:
                line.append(blob)
                assigned = True
                break
            dist = abs(blob["baseline"] - line_baseline)
            if dist < ah * LINE_ASSIGN_FRAC:
                line.append(blob)
                assigned = True
                break
        if not assigned:
            lines.append([blob])

    return lines


def _fit_baseline(blobs):
    if len(blobs) == 1:
        b = blobs[0]
        return b["baseline"], b["y1"], b["y2"]

    xs        = np.array([b["cx"]      for b in blobs], dtype=np.float64)
    baselines = np.array([b["baseline"] for b in blobs], dtype=np.float64)

    if xs.max() - xs.min() < 1:
        fitted = baselines
    else:
        coeffs = np.polyfit(xs, baselines, deg=1)
        fitted = np.polyval(coeffs, xs)

    baseline_y = float(np.mean(fitted))
    top_y      = float(min(b["y1"] for b in blobs))
    bot_y      = float(max(b["y2"] for b in blobs))
    return baseline_y, top_y, bot_y


def detect_lines(binary, ah=None, block_x=0, block_y=0):
    if ah is None:
        ah = _estimate_ah(binary)

    blobs = _extract_blobs(binary, ah)
    if not blobs:
        return [], ah

    raw_lines = _assign_blobs_to_lines(blobs, ah)
    raw_lines = [l for l in raw_lines if len(l) >= MIN_LINE_BLOBS]
    raw_lines.sort(key=lambda l: np.median([b["baseline"] for b in l]))

    pad      = max(1, int(ah * LINE_PAD_FRAC))
    img_h    = binary.shape[0]
    lines    = []

    for line_blobs in raw_lines:
        _, top_y, bot_y = _fit_baseline(line_blobs)
        ls = max(0,     int(top_y) - pad  + block_y)
        le = min(img_h + block_y, int(bot_y) + pad + block_y)
        lines.append((ls, le))

    return lines, ah