import numpy as np
import cv2
from skimage.feature import hog

def hog_feature(grayscale_patch):
    grayscale_patch = np.sqrt(grayscale_patch / 255.0)
    features = hog(
        grayscale_patch,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
        channel_axis=None,
    )
    return features.astype(np.float32)

def sift_descriptor_feature(grayscale_patch):
    x_gradient = cv2.Sobel(grayscale_patch, cv2.CV_32F, 1, 0, ksize=1)
    y_gradient = cv2.Sobel(grayscale_patch, cv2.CV_32F, 0, 1, ksize=1)
    magnitude = np.sqrt(x_gradient ** 2 + y_gradient ** 2)
    orientation = np.arctan2(np.abs(y_gradient), np.abs(x_gradient)) * (180.0 / np.pi)

    grid_size = 4
    bins = 8

    height, width = grayscale_patch.shape
    cell_height = height // grid_size
    cell_width = width // grid_size

    features = []
    for row in range(grid_size):
        for column in range(grid_size):
            x_start = column * cell_width
            y_start = row * cell_height
            x_end = (column + 1) * cell_width
            y_end = (row + 1) * cell_height

            cell_magnitude   = magnitude[y_start:y_end, x_start:x_end]
            cell_orientation = orientation[y_start:y_end, x_start:x_end]
            gauss = cv2.getGaussianKernel(cell_height, cell_height / 2.0) * cv2.getGaussianKernel(cell_width, cell_width / 2.0).T
            gauss = gauss / (gauss.max() + 1e-8)
            cell_magnitude = cell_magnitude * gauss
            histogram, _ = np.histogram(
                cell_orientation.ravel(), bins=bins, range=(0.0, 90.0),
                weights=cell_magnitude.ravel(),
            )

            histogram = histogram / (np.linalg.norm(histogram) + 1e-8)
            histogram = np.clip(histogram, 0.0, 0.2)
            histogram = histogram / (np.linalg.norm(histogram) + 1e-8)

            features.extend(histogram)

    return np.array(features, dtype=np.float32)

def canny_edge_map_feature(grayscale_patch):
    edges = cv2.Canny(grayscale_patch.astype(np.uint8), threshold1=50, threshold2=150)    
    grid_size = 8
    height, width  = edges.shape
    cell_height    = height // grid_size
    cell_width    = width // grid_size

    features = []
    for row in range(grid_size):
        for column in range(grid_size):
            x_start = column * cell_width
            y_start = row * cell_height
            x_end = (column + 1) * cell_width
            y_end = (row + 1) * cell_height

            cell   = edges[y_start:y_end, x_start:x_end]
            edge_density = float(np.mean(cell > 0))
            features.append(edge_density)

    return np.array(features, dtype=np.float32)

def extract_features(image_patch):
    resized_patch = cv2.resize(image_patch, (96, 96))
    if len(resized_patch.shape) == 3 and resized_patch.shape[2] == 3:
        grayscale_patch = cv2.cvtColor(resized_patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        grayscale_patch = resized_patch.astype(np.float32)
    hog = hog_feature(grayscale_patch)
    sift = sift_descriptor_feature(grayscale_patch)
    canny = canny_edge_map_feature(grayscale_patch)
    return np.concatenate([hog, sift, canny]).astype(np.float32)