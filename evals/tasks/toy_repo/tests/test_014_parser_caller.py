from toybox.caller import sum_parsed
from toybox.parser import parse_int_list


def test_parse_int_list_returns_ints():
    assert parse_int_list("1,2,3") == [1, 2, 3]


def test_sum_parsed():
    assert sum_parsed("1,2,3") == 6
    assert sum_parsed("10") == 10
