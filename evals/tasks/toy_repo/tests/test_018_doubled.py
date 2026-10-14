from toybox.mathutil import doubled


def test_doubled():
    assert doubled(3) == 6
    assert doubled(0) == 0
