from toybox.stats import median


def test_median_does_not_mutate():
    xs = [3, 1, 2]
    assert median(xs) == 2
    assert xs == [3, 1, 2]
