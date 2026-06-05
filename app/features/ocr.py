import io
import cv2
import numpy as np
from PIL import Image, ImageOps
from arabic_ocr.pipeline import ArabicOCRPipeline

_pipeline = None


def _get_pipeline() -> ArabicOCRPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = ArabicOCRPipeline(classifier="cnn", debug=True)
    return _pipeline


def _decode(image_bytes: bytes) -> np.ndarray:
    # Use Pillow to respect EXIF rotation (phone photos are often stored sideways)
    pil_img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes)))
    img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    return img


def handle(image_bytes: bytes) -> str:
    return _get_pipeline().run_array(_decode(image_bytes))
