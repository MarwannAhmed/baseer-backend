import cv2
import numpy as np

LINE_MERGE_TOLERANCE = 0.7
AH_HEIGHT_MIN        = 0.3
AH_HEIGHT_MAX        = 2.5
AH_AREA_MIN          = 0.05
LINE_PAD_FRACTION    = 0.15


def estimate_ah(binary):
    inv = 255 - binary
    n, _, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    heights = [stats[i][3] for i in range(1, n) if stats[i][4] > 5]
    return float(np.median(heights)) if heights else 20.0


def detect_lines(binary, ah=None):
    if ah is None:
        ah = estimate_ah(binary)

    inv = 255 - binary
    n, _, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)

    components = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if (AH_HEIGHT_MIN * ah < h < AH_HEIGHT_MAX * ah
                and area > AH_AREA_MIN * ah * ah):
            components.append((y + h / 2.0, y, y + h))

    if not components:
        return [], ah

    components.sort(key=lambda c: c[0])

    groups   = []
    g_cy_max = components[0][0]
    current  = [components[0]]

    for cy, y1, y2 in components[1:]:
        if cy - g_cy_max < ah * LINE_MERGE_TOLERANCE:
            current.append((cy, y1, y2))
            g_cy_max = max(g_cy_max, cy)
        else:
            groups.append(current)
            current  = [(cy, y1, y2)]
            g_cy_max = cy
    groups.append(current)

    pad    = max(1, int(ah * LINE_PAD_FRACTION))
    h_img  = binary.shape[0]
    lines  = []
    for group in groups:
        ls = max(0,     min(c[1] for c in group) - pad)
        le = min(h_img, max(c[2] for c in group) + pad)
        lines.append((ls, le))

    return lines, ah