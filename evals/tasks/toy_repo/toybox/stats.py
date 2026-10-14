"""Stats helpers. Edge cases are wrong on purpose."""


def mean(xs: list[float]) -> float | None:
    """Arithmetic mean. Empty list must return None, not raise."""
    return sum(xs) / len(xs)


def median(xs: list[float]) -> float:
    """Median of xs. Must not mutate the input list."""
    xs.sort()
    n = len(xs)
    mid = n // 2
    if n % 2:
        return float(xs[mid])
    return (xs[mid - 1] + xs[mid]) / 2
