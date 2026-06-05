from pathlib import Path
from app.core.image_utils import decode_image, resize_if_large
from app.features.color_detection.infer_color import load_model, extract_features

_MODELS_DIR = Path(__file__).resolve().parents[3] / "models" / "color_detection"
_model, _le = load_model(str(_MODELS_DIR / "color_svm.pkl"))


def handle(image_bytes: bytes) -> dict:
    bgr = decode_image(image_bytes)
    if bgr is None:
        return {"error": "Could not decode image"}

    bgr = resize_if_large(bgr)
    features, *_ = extract_features(bgr)

    label = _le.inverse_transform(_model.predict(features))[0]

    top3 = []
    confidence = 1.0
    if hasattr(_model, "predict_proba"):
        probs = _model.predict_proba(features)[0]
        top3_idx = probs.argsort()[::-1][:3]
        top3 = [
            {"color": _le.classes_[i], "confidence": round(float(probs[i]), 4)}
            for i in top3_idx
        ]
        confidence = top3[0]["confidence"] if top3 else 1.0

    return {
        "color": label,
        "confidence": confidence,
        "top3": top3,
    }
