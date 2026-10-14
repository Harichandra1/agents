import os

from toybox.config import get_with_env


def test_env_overrides_default(monkeypatch):
    monkeypatch.setenv("TOYBOX_RETRIES", "9")
    assert get_with_env("retries") == 9
    monkeypatch.delenv("TOYBOX_RETRIES")
    assert get_with_env("retries") == 3
