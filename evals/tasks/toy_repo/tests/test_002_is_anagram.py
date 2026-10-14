from toybox.textutil import is_anagram


def test_is_anagram():
    assert is_anagram("Listen", "Silent") is True
    assert is_anagram("hello", "world") is False
    assert is_anagram("a cat", "act a") is True
