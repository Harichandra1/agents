from toybox.mathutil import safe_div


def test_safe_div():
    assert safe_div(10, 2) == 5
    assert safe_div(1, 0) is None
