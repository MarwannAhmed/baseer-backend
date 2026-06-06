from unittest.mock import patch
from app.services.command_router import route_command, COMMAND_MAP


def test_command_map_contains_all_commands():
    assert "كشف" in COMMAND_MAP
    assert "نص"  in COMMAND_MAP
    assert "لون" in COMMAND_MAP


def test_unknown_command_returns_error():
    result = route_command("مجهول", b"")
    assert "error" in result


def test_unknown_command_error_lists_available_commands():
    result = route_command("xyz", b"")
    assert "كشف" in result["error"]
    assert "نص"  in result["error"]
    assert "لون" in result["error"]


def test_known_command_calls_correct_handler():
    dummy_bytes = b"image"
    with patch.dict("app.services.command_router.COMMAND_MAP", {"كشف": lambda b: {"objects": []}}):
        result = route_command("كشف", dummy_bytes)
    assert result == {"objects": []}


def test_handler_exception_returns_generic_error():
    def _crash(b):
        raise RuntimeError("boom")

    with patch.dict("app.services.command_router.COMMAND_MAP", {"كشف": _crash}):
        result = route_command("كشف", b"")
    assert "error" in result


def test_handler_not_implemented_returns_wip_error():
    def _wip(b):
        raise NotImplementedError

    with patch.dict("app.services.command_router.COMMAND_MAP", {"كشف": _wip}):
        result = route_command("كشف", b"")
    assert "error" in result
    assert "قيد التطوير" in result["error"]


def test_route_command_passes_bytes_to_handler():
    received = {}

    def _capture(b):
        received["bytes"] = b
        return {}

    dummy = b"\x01\x02\x03"
    with patch.dict("app.services.command_router.COMMAND_MAP", {"نص": _capture}):
        route_command("نص", dummy)

    assert received["bytes"] == dummy
