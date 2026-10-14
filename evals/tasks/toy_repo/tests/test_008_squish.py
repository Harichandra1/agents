from toybox.textutil import squish


def test_squish():
    assert squish("  hello   world  ") == "hello world"
    assert squish("a\n\tb") == "a b"
