import cv2
import numpy as np

from arabic_ocr.config import NORM_SIZE


def normalize(character_image: np.ndarray) -> np.ndarray:

    if character_image.size == 0:  
        return np.full((NORM_SIZE, NORM_SIZE), 255, dtype=np.uint8) 

    height, width = character_image.shape[:2]
    if height == 0 or width == 0:
        return np.full((NORM_SIZE, NORM_SIZE), 255, dtype=np.uint8)

    target = NORM_SIZE - 4  
    scale = target / max(height, width)
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR 
    resized = cv2.resize(character_image, (new_width, new_height), interpolation=interpolation)

    output_canvas = np.full((NORM_SIZE, NORM_SIZE), 255, dtype=np.uint8)
    y_offset = (NORM_SIZE - new_height) // 2 
    x_offset = (NORM_SIZE - new_width) // 2
    output_canvas[y_offset: y_offset + new_height, x_offset: x_offset + new_width] = resized

    return output_canvas
