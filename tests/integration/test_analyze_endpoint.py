from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

DUMMY_IMAGE = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
    b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
    b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1e"
    b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
    b"\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
    b"\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04"
    b"\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa"
    b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xff\xd9"
)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_detect_objects_returns_objects_key():
    r = client.post(
        "/analyze",
        data={"command": "كشف"},
        files={"file": ("test.jpg", DUMMY_IMAGE, "image/jpeg")},
    )
    assert r.status_code == 200
    assert "objects" in r.json()


def test_extract_text_returns_description_key():
    r = client.post(
        "/analyze",
        data={"command": "نص"},
        files={"file": ("test.jpg", DUMMY_IMAGE, "image/jpeg")},
    )
    assert r.status_code == 200
    assert "description" in r.json()


def test_detect_color_returns_color_key():
    r = client.post(
        "/analyze",
        data={"command": "لون"},
        files={"file": ("test.jpg", DUMMY_IMAGE, "image/jpeg")},
    )
    assert r.status_code == 200
    assert "color" in r.json()


def test_unknown_command_returns_error():
    r = client.post(
        "/analyze",
        data={"command": "مجهول"},
        files={"file": ("test.jpg", DUMMY_IMAGE, "image/jpeg")},
    )
    assert r.status_code == 200
    assert "error" in r.json().get("description", r.json())


def test_empty_file_returns_400():
    r = client.post(
        "/analyze",
        data={"command": "كشف"},
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert r.status_code == 400
