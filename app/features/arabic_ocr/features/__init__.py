import numpy as np
from arabic_ocr.segment.dots import Dot
from .normalize import normalize
from .grid_density import grid_density
from .hog_features import hog_features
from .contour_features import contour_features
from .zernike_moments import zernike_features
from .dot_features import dot_features


def extract(char_img, dot_list = None):
    norm = normalize(char_img)
    parts = [
        grid_density(norm),          
        hog_features(norm),          
        contour_features(norm),      
        zernike_features(norm),      
        dot_features(dot_list),      
    ]
    return np.concatenate(parts).astype(np.float32)


def extract_batch(char_imgs,dot_lists = None):
    if dot_lists is None:
        dot_lists = [None] * len(char_imgs)
    vectors = [extract(img, dots) for img, dots in zip(char_imgs, dot_lists)]
    return np.stack(vectors, axis=0) if vectors else np.empty((0,), dtype=np.float32)


__all__ = ["extract", "extract_batch"]
