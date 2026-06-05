from app.core.image_utils import decode_image
from app.features.text_extraction.dl.inference import infer


def run_extraction(image_bytes: bytes, det_model=None, rec_model=None):
    img = decode_image(image_bytes)

    return infer(img, det_model=det_model, rec_model=rec_model)