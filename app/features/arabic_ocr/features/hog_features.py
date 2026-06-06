import numpy as np
from skimage.feature import hog


def hog_features(normalized_img: np.ndarray):

    feature_descriptor = hog(
        normalized_img,
        orientations=9, 
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2), 
        block_norm="L2-Hys", 
        visualize=False,
        feature_vector=True, 
    )
    return feature_descriptor.astype(np.float32)
