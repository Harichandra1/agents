from toybox.stats import mean


def test_mean():
    assert mean([2, 4, 6]) == 4
    assert mean([]) is None
