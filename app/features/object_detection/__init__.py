from app.features.object_detection.predict import run_detection


def handle(image_bytes: bytes) -> list:
    return run_detection(image_bytes)
