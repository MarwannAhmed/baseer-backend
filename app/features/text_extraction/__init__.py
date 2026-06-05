from app.features.text_extraction.pipeline import run_extraction
from app.features.text_extraction.pipeline_dl import run_extraction as run_extraction_dl
from app.features.text_extraction.dl.inference import load_models

det_model = None
rec_model = None

def handle(image_bytes: bytes, run_classical: bool = False):
    global det_model, rec_model
    if det_model is None or rec_model is None:
        det_model, rec_model = load_models()
    if run_classical:
        return run_extraction(image_bytes)
    else:
        return run_extraction_dl(image_bytes, det_model, rec_model)