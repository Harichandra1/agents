"""
Arithmetic tools used by the Day 1 bare-loop exercise.

Deliberately trivial: the point of Day 1 is the loop mechanics, not the tools.
"""
from langchain_core.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Add `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` by `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b


ARITHMETIC_TOOLS = [add, multiply, divide]
