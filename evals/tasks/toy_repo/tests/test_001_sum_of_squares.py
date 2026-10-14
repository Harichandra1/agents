from toybox.mathutil import sum_of_squares


def test_sum_of_squares():
    assert sum_of_squares([1, 2, 3, 4]) == 30
    assert sum_of_squares([]) == 0
