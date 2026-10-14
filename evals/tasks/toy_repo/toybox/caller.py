"""Uses parser.parse_int_list. Breaks until the parser returns ints."""

from toybox.parser import parse_int_list


def sum_parsed(s: str) -> int:
    """Sum the integers in a comma-separated string."""
    return sum(parse_int_list(s))
