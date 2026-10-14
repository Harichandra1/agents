from toybox.config import get


def test_get_missing_key_returns_none():
    assert get("retries") == 3
    assert get("nope") is None
