import cv2
import numpy as np
from arabic_ocr.pipeline import ArabicOCRPipeline

_pipeline = None


def _get_pipeline() -> ArabicOCRPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = ArabicOCRPipeline(classifier="cnn", debug=True)
    return _pipeline


def handle(image_bytes: bytes) -> str:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return _get_pipeline().run_array(img)
