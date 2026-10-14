from toybox.textutil import fizzbuzz_counts


def test_fizzbuzz_counts():
    fizz, buzz, fizzbuzz = fizzbuzz_counts(30)
    assert fizz == 8
    assert buzz == 4
    assert fizzbuzz == 2
