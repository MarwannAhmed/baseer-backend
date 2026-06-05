import numpy as np
import cv2


def decode_image(raw_image: bytes):
    byte_array = np.frombuffer(raw_image, np.uint8)
    image = cv2.imdecode(byte_array, cv2.IMREAD_COLOR)
    return image

def resize_if_large(image, max_dim: int = 1280):
    height, width = image.shape[:2]
    dimension = max(height, width)
    if dimension <= max_dim:
        return image
    else:
        scale_factor = max_dim / dimension
        scaled_width = int(width * scale_factor)
        scaled_height = int(height * scale_factor)
        return cv2.resize(image, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)