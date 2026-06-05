from app.features.object_detection import handle as detect_objects
from app.features.ocr import handle as extract_text
from app.features.color_detection import handle as detect_color

# ── Register commands here ────────────────────────────────────────────────────

COMMAND_MAP: dict = {
    "كشف": detect_objects,
    "نص":  extract_text,
    "لون": detect_color,
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