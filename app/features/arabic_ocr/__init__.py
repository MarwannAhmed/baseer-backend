from arabic_ocr.pipeline import ArabicOCRPipeline
from app.core.image_utils import decode_image

_pipeline = ArabicOCRPipeline(classifier="cnn", debug=True)


def handle(image_bytes: bytes) -> dict:
    img = decode_image(image_bytes)
    if img is None:
        return {"error": "Could not decode image"}

    text = _pipeline.run_array(img)
    return {"text": text}


__all__ = ["ArabicOCRPipeline", "handle"]
