import numpy as np
import cv2


def decode_image(image_bytes: bytes):
    """
    Decode raw image bytes (JPEG, PNG, etc.) into a BGR numpy array.

    Parameters
    ----------
    image_bytes : bytes  — raw file contents from the upload

    Returns
    -------
    np.ndarray  BGR image, or None if decoding failed
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


def resize_if_large(image, max_dim: int = 1280):
    """
    Downscale the image if either dimension exceeds max_dim.
    Keeps aspect ratio. Helps keep inference time predictable.

    Parameters
    ----------
    image   : np.ndarray  BGR image
    max_dim : int         maximum allowed width or height

    Returns
    -------
    np.ndarray  original or resized image
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image

    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)