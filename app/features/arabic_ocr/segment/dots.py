import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class Dot:
    cx: float
    cy: float
    cluster_size: int          # number of dot blobs in this cluster
    position: str              


def separate_dots(paw_binary, ah):
    inverted = cv2.bitwise_not(paw_binary)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    baseline_y = _estimate_baseline(stats, labels, num_labels, ah)
    body = inverted.copy()
    raw_dots: list[tuple[float, float]] = []

    heights = []
    widths = []
    areas = []
    for label in range(1, num_labels):
        height = stats[label, cv2.CC_STAT_HEIGHT]
        width = stats[label, cv2.CC_STAT_WIDTH]
        area = stats[label, cv2.CC_STAT_AREA]
        heights.append(height)
        widths.append(width)
        areas.append(area)

    heights = np.array(heights) if heights else np.array([ah])
    widths = np.array(widths) if widths else np.array([ah])
    areas = np.array(areas) if areas else np.array([ah * ah])

    # Adaptive thresholds based on component size distribution and ah
    h_thresh = max(1.0, min(0.6 * ah, float(np.percentile(heights, 25) * 1.25)))
    w_thresh = max(1.0, min(0.6 * ah, float(np.percentile(widths, 25) * 1.25)))
    area_thresh = max(1.0, min(0.2 * ah * ah, float(np.percentile(areas, 25) * 1.5)))
    aspect_min, aspect_max = 0.2, 6.0

    candidate_lbls = set()
    max_dot_dim = max(2.0, 0.6 * ah)
    for label in range(1, num_labels):
        height = stats[label, cv2.CC_STAT_HEIGHT]
        width = stats[label, cv2.CC_STAT_WIDTH]
        area = stats[label, cv2.CC_STAT_AREA]
        cx, cy = centroids[label]
        aspect = height / (width + 1e-5)
        small_dim = max(height, width) <= max_dot_dim
        small_area = area <= area_thresh
        if not (small_dim or small_area):
            continue
        if not (aspect_min <= aspect <= aspect_max):
            continue
        left = int(stats[label, cv2.CC_STAT_LEFT])
        top = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        inv_bbox = inverted[top:top+bh, left:left+bw]
        lbl_bbox = labels[top:top+bh, left:left+bw]
        other_ink = np.any((inv_bbox > 0) & (lbl_bbox != label))
        if other_ink:
            continue
        candidate_lbls.add(label)

    for label in sorted(candidate_lbls):
        body[labels == label] = 0  
        cx, cy = centroids[label]
        raw_dots.append((cx, cy))

    body_binary = cv2.bitwise_not(body)

    # Cluster nearby dots (within 0.5*ah)
    dot_list = _cluster_dots(raw_dots, baseline_y, cluster_radius=0.5 * ah)

    if len(dot_list) > 8:
        import logging
        logging.getLogger(__name__).warning("Unusually many dots detected: %d", len(dot_list))

    return body_binary, dot_list


def _estimate_baseline(stats, labels, num_labels, ah):
    bottoms = []
    for label in range(1, num_labels):
        height = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]
        top  = stats[label, cv2.CC_STAT_TOP]
        width = stats[label, cv2.CC_STAT_WIDTH]
        aspect = height / (width + 1e-5)
        if not (height < 0.4 * ah and width < 0.4 * ah and area < 0.15 * ah * ah
                and 0.25 <= aspect <= 4.0):
            bottoms.append(top + height)
    if not bottoms:
        return float(labels.shape[0]) * 0.75
    values, counts = np.unique(bottoms, return_counts=True)
    return float(values[np.argmax(counts)])


def _cluster_dots(raw_dots, baseline_y, cluster_radius):
    if not raw_dots:
        return []

    n = len(raw_dots)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if np.hypot(raw_dots[i][0] - raw_dots[j][0],
                        raw_dots[i][1] - raw_dots[j][1]) <= cluster_radius:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    groups: dict[int, list[tuple[float, float]]] = {}
    for i, pt in enumerate(raw_dots):
        groups.setdefault(find(i), []).append(pt)

    clusters: list[Dot] = []
    for group in groups.values():
        mean_cx = float(np.mean([p[0] for p in group]))
        mean_cy = float(np.mean([p[1] for p in group]))
        position = "above" if mean_cy < baseline_y else "below"
        clusters.append(Dot(cx=mean_cx, cy=mean_cy,
                            cluster_size=len(group), position=position))
    return clusters
