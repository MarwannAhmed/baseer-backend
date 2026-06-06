from pyexpat import features

import numpy as np

from arabic_ocr.config import GRID_ROWS, GRID_COLS, NORM_SIZE


def grid_density(normalized_img: np.ndarray):

    assert normalized_img.shape == (NORM_SIZE, NORM_SIZE), (
        f"Expected {NORM_SIZE}×{NORM_SIZE}, got {normalized_img.shape}" ) 

    cell_height = NORM_SIZE // GRID_ROWS  
    cell_width = NORM_SIZE // GRID_COLS  
    cell_area = cell_height * cell_width

    features = np.empty(GRID_ROWS * GRID_COLS, dtype=np.float32)
    for r in range(GRID_ROWS):   
        for c in range(GRID_COLS):
            cell = normalized_img[ r * cell_height: (r + 1) * cell_height, c * cell_width: (c + 1) * cell_width]
            features[r * GRID_COLS + c] = float(np.sum(cell == 0)) / cell_area  

    return features