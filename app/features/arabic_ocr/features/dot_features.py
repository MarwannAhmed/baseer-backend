import numpy as np
from arabic_ocr.segment.dots import Dot


def dot_features(dot_list: list[Dot] | None):
    if not dot_list:
        return np.zeros(4, dtype=np.float32)

    above = sum(d.cluster_size for d in dot_list if d.position == "above")
    below = sum(d.cluster_size for d in dot_list if d.position == "below")
    has_dot = 1.0 if dot_list else 0.0

    xs = [d.cx for d in dot_list] 
    horizontal_spread = float(np.std(xs)) if len(xs) > 1 else 0.0 #high spread if close

    return np.array([above, below, has_dot, horizontal_spread], dtype=np.float32)
