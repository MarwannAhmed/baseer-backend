from app.features.object_detection import handle as detect_objects
from app.features.color_detection import handle as detect_color
from app.features.text_extraction import handle as extract_text
from app.features.arabic_ocr import handle as extract_text_ocr_ar
from app.features.text_extraction.pipeline import run_extraction as extract_text_ocr

COMMAND_MAP: dict = {
    "كشف": detect_objects,
    "نص":  extract_text,
    "لون": detect_color,
    "نصا": extract_text_ocr,
    "نصعر": extract_text_ocr_ar,
}

def route_command(command: str, image_bytes: bytes):

    handler = COMMAND_MAP.get(command)

    if handler is None:
        available = "، ".join(COMMAND_MAP.keys())
        return {"error": f"الأمر '{command}' غير معروف. الأوامر المتاحة: {available}"}

    try:
        return handler(image_bytes)
    except NotImplementedError:
        return {"error": "هذه الميزة قيد التطوير"}
    except Exception as e:
        print(f"[ERROR] command='{command}': {e}")
        return {"error": "حدث خطأ أثناء المعالجة، يرجى المحاولة مرة أخرى"}