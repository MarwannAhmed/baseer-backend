import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from app.features.text_extraction.config import parameters


def _estimate_ah(binary):
    inv = 255 - binary
    n, _, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    heights = [stats[i][3] for i in range(1, n) if stats[i][4] > 5]
    return float(np.median(heights)) if heights else 20.0

def detect_lines(binary, ah=None):
    if ah is None:
        ah = _estimate_ah(binary)

    inv = 255 - binary
    n, _, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)

    components = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if (parameters["average"]["height_min"] * ah < h < parameters["average"]["height_max"] * ah
                and area > parameters["average"]["area_min"] * ah * ah):
            components.append((y + h / 2.0, y, y + h))

    if not components:
        return [], ah

    h_img  = binary.shape[0]
    votes  = np.zeros(h_img, dtype=float)
    sigma  = ah * parameters["lines"]["kde_bandwidth"]

    for cy, _, _ in components:
        votes[int(np.clip(cy, 0, h_img - 1))] += 1.0

    density  = gaussian_filter1d(votes, sigma=sigma)
    min_dist = int(ah * 0.8)
    if min_dist == 0:
        min_dist = 1
    peaks, _ = find_peaks(density,
                          distance=min_dist,
                          height=density.max() * parameters["lines"]["peak_prominence"])

    if len(peaks) == 0:
        return [], ah

    max_dist  = ah * parameters["lines"]["max_assign_dist"]
    groups    = {p: [] for p in peaks}
    orphans   = []

    for comp in components:
        cy, y1, y2 = comp
        dists      = np.abs(peaks - cy)
        nearest_i  = int(np.argmin(dists))
        if dists[nearest_i] <= max_dist:
            groups[peaks[nearest_i]].append(comp)
        else:
            orphans.append(comp)

    orphan_lines = []
    if orphans:
        orphans.sort(key=lambda c: c[0])
        cluster, cluster_y2 = [orphans[0]], orphans[0][2]
        for comp in orphans[1:]:
            if comp[1] <= cluster_y2 + ah * 0.5:
                cluster.append(comp)
                cluster_y2 = max(cluster_y2, comp[2])
            else:
                if len(cluster) >= parameters["lines"]["orphan_min_members"]:
                    orphan_lines.append(cluster)
                cluster, cluster_y2 = [comp], comp[2]
        if len(cluster) >= parameters["lines"]["orphan_min_members"]:
            orphan_lines.append(cluster)

    pad      = max(1, int(ah * parameters["lines"]["line_pad_fraction"]))
    all_lines = []

    for members in groups.values():
        if not members:
            continue
        ls = max(0,     min(c[1] for c in members) - pad)
        le = min(h_img, max(c[2] for c in members) + pad)
        all_lines.append((ls, le))

    for members in orphan_lines:
        ls = max(0,     min(c[1] for c in members) - pad)
        le = min(h_img, max(c[2] for c in members) + pad)
        all_lines.append((ls, le))

    all_lines.sort(key=lambda l: l[0])
    return all_lines, ah