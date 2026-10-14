from toybox.mathutil import palindrome


def test_palindrome():
    assert palindrome("aba") is True
    assert palindrome("ab") is False
    assert palindrome("") is True
