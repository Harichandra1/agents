"""String helpers. Several functions are missing or wrong on purpose."""


def reverse(s: str) -> str:
    return s[::-1]


def squish(s: str) -> str:
    """Collapse internal whitespace to single spaces and strip the ends."""
    import re

    return re.sub(r"\s+", " ", s)


def title_case(s: str) -> str:
    """Capitalize the first letter of each whitespace-separated word."""
    return s.upper()
