"""Tiny config. Missing-key and env-override behavior is incomplete."""

DEFAULTS = {"retries": 3, "mode": "fast"}


def get(key: str):
    """Return DEFAULTS[key]. Unknown keys must return None, not KeyError."""
    return DEFAULTS[key]


def get_with_env(key: str):
    """Like get, but TOYBOX_{KEY} in the environment wins when set."""
    return DEFAULTS[key]
