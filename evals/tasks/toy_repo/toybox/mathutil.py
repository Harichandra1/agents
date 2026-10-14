"""Numeric helpers. Several functions are missing or wrong on purpose."""


def add(a: int, b: int) -> int:
    return a + b


def inclusive_range(start: int, end: int) -> list[int]:
    """Integers from start to end inclusive."""
    return list(range(start, end))


def safe_div(a: float, b: float) -> float | None:
    """Divide a by b. Return None when b is 0."""
    return a / b


def palindrome(s: str) -> bool:
    """True if s equals its reverse."""
    return s == s
