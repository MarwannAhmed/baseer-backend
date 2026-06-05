from app.features.object_detection import handle as detect_objects
from app.features.text_extraction import handle as extract_text

# ── Register commands here ────────────────────────────────────────────────────

COMMAND_MAP: dict = {
    "كشف":   detect_objects,     
    # "مسافة": estimate_distance,  
    "نص":    extract_text,
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