import cv2
import numpy as np
from app.features.color_detection import handle


def _encode(bgr_tuple, size=64) -> bytes:
    img = np.full((size, size, 3), bgr_tuple, dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


RED   = _encode((0,   0,   200))
BLUE  = _encode((200, 0,   0))
WHITE = _encode((255, 255, 255))
BLACK = _encode((0,   0,   0))


def test_handle_returns_color_key():
    result = handle(RED)
    assert "color" in result


def test_handle_returns_confidence_key():
    result = handle(RED)
    assert "confidence" in result


def test_handle_returns_top3_key():
    result = handle(RED)
    assert "top3" in result


def test_handle_color_is_string():
    result = handle(RED)
    assert isinstance(result["color"], str)


def test_handle_confidence_in_range():
    result = handle(RED)
    assert 0.0 <= result["confidence"] <= 1.0


def test_handle_top3_at_most_three():
    result = handle(RED)
    assert len(result["top3"]) <= 3


def test_handle_top3_confidences_are_valid():
    result = handle(RED)
    for entry in result["top3"]:
        assert "color" in entry
        assert "confidence" in entry
        assert 0.0 <= entry["confidence"] <= 1.0


def test_handle_top3_sorted_descending():
    result = handle(RED)
    confs = [e["confidence"] for e in result["top3"]]
    assert confs == sorted(confs, reverse=True)


def test_handle_top3_first_matches_prediction():
    result = handle(RED)
    if result["top3"]:
        assert result["top3"][0]["color"] == result["color"]


def test_handle_invalid_bytes_returns_error():
    result = handle(b"not an image")
    assert "error" in result


def test_handle_empty_bytes_returns_error():
    result = handle(b"")
    assert "error" in result


def test_handle_white_image():
    result = handle(WHITE)
    assert "color" in result
    assert isinstance(result["color"], str)


def test_handle_black_image():
    result = handle(BLACK)
    assert "color" in result
    assert isinstance(result["color"], str)


def test_handle_blue_image():
    result = handle(BLUE)
    assert "color" in result
    assert isinstance(result["color"], str)
