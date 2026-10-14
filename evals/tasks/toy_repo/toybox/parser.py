"""Parse a comma-separated list of integers. Returns strings today — a bug."""


def parse_int_list(s: str) -> list[int]:
    """'1,2,3' -> [1, 2, 3]. Empty pieces should be skipped."""
    return s.split(",")
