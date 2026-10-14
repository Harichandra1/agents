from toybox.csvparse import parse_rows


def test_parse_rows_skips_header():
    text = "name,age\nalice,3\nbob,4\n"
    assert parse_rows(text) == [["alice", "3"], ["bob", "4"]]
