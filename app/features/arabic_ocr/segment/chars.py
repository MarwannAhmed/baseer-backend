import logging
import numpy as np
from arabic_ocr.config import AH_HEIGHT_MIN, AH_HEIGHT_MAX,MIN_CHAR_WIDTH, MAX_CHAR_WIDTH, AH_AREA_MIN,CHOP_MIN_VALLEY
import cv2

logger = logging.getLogger(__name__)

def _segment_chars_impl(paw_binary, paw_x, paw_y, ah):
    try:
        column_proj = np.sum(paw_binary == 0, axis=0).astype(float)
        height, width = paw_binary.shape

        if width < 2 or column_proj.max() == 0:
            return [(paw_x, paw_y, paw_x + width, paw_y + height)]

        # Find valleys as midpoints of contiguous low-ink regions.
        local_max = np.percentile(column_proj[column_proj > 0], 90) if column_proj.any() else 1.0
        threshold = CHOP_MIN_VALLEY * local_max

        # Minimum gap width: scale with average character height (ah).
        # Use AH-based sizing so the same code works across DPIs and scans.
        min_gap_width = max(1, int(round(0.06 * ah)))

        valleys: list[int] = []
        in_gap = False
        gap_start = 0
        for i, v in enumerate(column_proj):
            if v < threshold and not in_gap:
                gap_start = i
                in_gap = True
            elif v >= threshold and in_gap:
                if (i - gap_start) >= min_gap_width:
                    valleys.append((gap_start + i - 1) // 2)
                in_gap = False
        if in_gap and (width - gap_start) >= min_gap_width:
            valleys.append((gap_start + width - 1) // 2)

        if not valleys:
            return [(paw_x, paw_y, paw_x + width, paw_y + height)]

        cut_candidates = sorted(set([0] + valleys + [width]))

        try:
            best_cuts = _best_segmentation(
                paw_binary, cut_candidates, ah,
                paw_h=height, paw_w=width,
            )
        except Exception:
            logger.exception("_best_segmentation failed, returning full PAW as one character")
            return [(paw_x, paw_y, paw_x + width, paw_y + height)]

        result = []
        for c1, c2 in zip(best_cuts[:-1], best_cuts[1:]):
            if c2 - c1 < 1:
                continue
            char_crop = paw_binary[:, c1:c2]
            char_h, char_w = char_crop.shape
            result.append((
                paw_x + c1, paw_y,
                paw_x + c2, paw_y + char_h,
            ))

        result.sort(key=lambda t: t[0], reverse=True)
        if not result:
            return [(paw_x, paw_y, paw_x + width, paw_y + height)]

        # Merge any accidentally tiny slivers into neighbours.
        min_char_w = max(2, int(round(0.25 * ah)))
        sliver_thresh = min_char_w
        cleaned: list[tuple[int, int, int, int]] = []
        for idx, box in enumerate(result):
            x1, y1, x2, y2 = box
            w_box = x2 - x1
            if w_box <= sliver_thresh:
                # merge small sliver into previous if exists, else into next
                if cleaned:
                    px1, py1, px2, py2 = cleaned[-1]
                    cleaned[-1] = (px1, py1, x2, py2)
                else:
                    if idx + 1 < len(result):
                        nx1, ny1, nx2, ny2 = result[idx + 1]
                        cleaned.append((x1, y1, nx2, y2))
                    else:
                        cleaned.append(box)
            else:
                cleaned.append(box)

        # If some segments look invalid, attempts to repair by merging adjacent segments and re-checking validity.
        def all_valid(segments):
            for sx1, sy1, sx2, sy2 in segments:
                crop = paw_binary[:, sx1 - paw_x:sx2 - paw_x]
                if not _valid_char(crop, ah):
                    return False
            return True

        repaired = list(cleaned)
        max_iters = len(repaired) * 2
        it = 0
        while it < max_iters and not all_valid(repaired) and len(repaired) > 1:
            for i, (sx1, sy1, sx2, sy2) in enumerate(repaired):
                crop = paw_binary[:, sx1 - paw_x:sx2 - paw_x]
                if not _valid_char(crop, ah):
                    if i > 0:
                        px1, py1, px2, py2 = repaired[i - 1]
                        repaired[i - 1] = (px1, py1, sx2, py2)
                        repaired.pop(i)
                    elif i + 1 < len(repaired):
                        nx1, ny1, nx2, ny2 = repaired[i + 1]
                        repaired[i + 1] = (sx1, sy1, nx2, ny2)
                        repaired.pop(i)
                    break
            it += 1

        # If we still have too many fragments, merge the smallest gaps until the PAW has a reasonable number of character boxes
        max_segments = max(2, int(round(width / max(1.0, 0.9 * ah))))
        if len(repaired) > max_segments:
            repaired = sorted(repaired, key=lambda t: t[0])
            while len(repaired) > max_segments:
                gap_info = []
                for i in range(len(repaired) - 1):
                    left = repaired[i]
                    right = repaired[i + 1]
                    gap = right[0] - left[2]
                    gap_info.append((gap, i))
                _, merge_idx = min(gap_info, key=lambda t: t[0])
                lx1, ly1, lx2, ly2 = repaired[merge_idx]
                rx1, ry1, rx2, ry2 = repaired[merge_idx + 1]
                repaired[merge_idx] = (
                    lx1,
                    min(ly1, ry1),
                    rx2,
                    max(ly2, ry2),
                )
                repaired.pop(merge_idx + 1)
            repaired.sort(key=lambda t: t[0], reverse=True)

        # Heuristic: if many returned segments are tiny (width << AH), merges adjacent tiny ones 
        tiny_thresh = max(1, int(round(0.25 * ah)))
        tiny_segments = [1 for x1, y1, x2, y2 in repaired if (x2 - x1) <= tiny_thresh]
        TINY_FRACTION_LIMIT = 0.25
        if repaired and (sum(tiny_segments) / len(repaired)) > TINY_FRACTION_LIMIT:
            # merges adjacent tiny segments first, then merges nearest neighbors by smallest gap.
            merged = list(sorted(repaired, key=lambda t: t[0]))
            iter_limit = len(merged) * 3
            iters = 0
            while iters < iter_limit:
                iters += 1
                tiny_count = sum(1 for a, b, c, d in merged if (c - a) <= tiny_thresh)
                if tiny_count / len(merged) <= TINY_FRACTION_LIMIT or len(merged) <= 1:
                    break
                did_merge = False
                i = 0
                while i < len(merged) - 1:
                    a1, b1, c1, d1 = merged[i]
                    a2, b2, c2, d2 = merged[i + 1]
                    w1 = c1 - a1
                    w2 = c2 - a2
                    if w1 <= tiny_thresh or w2 <= tiny_thresh:
                        merged[i] = (a1, min(b1, b2), c2, max(d1, d2))
                        merged.pop(i + 1)
                        did_merge = True
                    else:
                        i += 1
                if did_merge:
                    continue
                if len(merged) > 1:
                    gap_info = []
                    for i in range(len(merged) - 1):
                        left = merged[i]
                        right = merged[i + 1]
                        gap = right[0] - left[2]
                        gap_info.append((gap, i))
                    _, merge_idx = min(gap_info, key=lambda t: t[0])
                    lx1, ly1, lx2, ly2 = merged[merge_idx]
                    rx1, ry1, rx2, ry2 = merged[merge_idx + 1]
                    merged[merge_idx] = (
                        lx1,
                        min(ly1, ry1),
                        rx2,
                        max(ly2, ry2),
                    )
                    merged.pop(merge_idx + 1)
                else:
                    break
            repaired = merged

        if all_valid(repaired):
            return repaired

        # If repair failed, fallback to whole PAW
        return [(paw_x, paw_y, paw_x + width, paw_y + height)]

    except Exception:
        logger.exception("segment_chars failed, returning full PAW")
        height, width = paw_binary.shape
        return [(paw_x, paw_y, paw_x + width, paw_y + height)]


def segment_chars(paw_binary, paw_x, paw_y, ah):
    return _segment_chars_impl(paw_binary, paw_x, paw_y, ah)


def _valid_char(crop, ah):
    if not np.any(crop == 0):
        return False
    #counts rows that actually contain ink
    tight_h = int(np.sum(np.any(crop == 0, axis=1)))
    _, width = crop.shape
    area = int(np.sum(crop == 0))
    return (
        AH_HEIGHT_MIN * ah <= tight_h <= AH_HEIGHT_MAX * ah and
        MIN_CHAR_WIDTH * ah <= width <= MAX_CHAR_WIDTH * ah and
        area >= AH_AREA_MIN * ah * ah
    )


def _best_segmentation(paw_binary, cuts, ah, paw_h, paw_w):
    n = len(cuts)
    if n <= 2:
        return cuts

    dp: list[tuple[int, int, int]] = [(-1, 0, -1)] * n
    dp[0] = (0, 0, -1)

    for i in range(1, n):
        for j in range(i):
            if dp[j][0] < 0:
                continue
            valid = _valid_char(paw_binary[:, cuts[j]:cuts[i]], ah)
            score = dp[j][0] + (1 if valid else 0)
            segments  = dp[j][1] + 1

            effective = score * 100 - segments * 12
            current_effective = dp[i][0] * 100 - dp[i][1] * 12 if dp[i][0] >= 0 else -10_000
            if effective > current_effective:
                dp[i] = (score, segments, j)

    # If no valid character was ever found, return the whole PAW unsplit.
    if dp[n - 1][0] <= 0:
        return [0, paw_w]

    path: list[int] = []
    i = n - 1
    while i >= 0:
        path.append(cuts[i])
        i = dp[i][2]

    return sorted(path)


def _estimate_ah(line_binary):
    inverted = cv2.bitwise_not(line_binary)
    _, _, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    if stats.shape[0] <= 1:
        return 20.0
    heights = stats[1:, cv2.CC_STAT_HEIGHT].astype(float)
    return float(np.percentile(heights, 70))


def segment_chars_with_fallback(paw_binary, paw_x, paw_y, ah):
    paw_height, paw_width = paw_binary.shape

    def score_segments(segments):
        if not segments:
            return -1.0
        valids = 0
        tiny = 0
        for x1, y1, x2, y2 in segments:
            crop = paw_binary[:, x1 - paw_x:x2 - paw_x]
            if _valid_char(crop, ah):
                valids += 1
            if (x2 - x1) <= max(1, int(round(0.25 * ah))):
                tiny += 1
        expected = max(1, int(round(paw_width / max(1.0, 0.9 * ah))))
        # prefers more valids, penalises tiny fraction and deviation from expected
        tiny_frac = tiny / len(segments)
        dev = abs(len(segments) - expected) / max(1, expected)
        return valids - tiny_frac * 0.5 - dev * 0.3

    candidates = []
    tried = set()
    # AH multipliers to try (centered at 1.0)
    multipliers = [1.0, 0.85, 1.15, 0.7, 1.3]
    for m in multipliers:
        scaled_ah = max(1.0, ah * m)
        key = int(round(scaled_ah))
        if key in tried:
            continue
        tried.add(key)
        try:
            segments = segment_chars(paw_binary, paw_x, paw_y, scaled_ah) if False else None
        except Exception:
            segments = None
        try:
            segments = _segment_chars_impl(paw_binary, paw_x, paw_y, scaled_ah)
        except Exception:
            segments = None
        if segments is None:
            continue
        candidates.append((score_segments(segments), segments))

    if not candidates:
        return [(paw_x, paw_y, paw_x + paw_width, paw_y + paw_height)]

    # picks best-scoring segmentation
    candidates.sort(key=lambda t: t[0], reverse=True)
    best = candidates[0][1]
    return best
