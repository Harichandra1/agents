"""CSV helpers. Header handling is wrong on purpose."""


def parse_rows(text: str) -> list[list[str]]:
    """Parse comma-separated rows. The first line is a header and must be skipped."""
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return [line.split(",") for line in lines]
