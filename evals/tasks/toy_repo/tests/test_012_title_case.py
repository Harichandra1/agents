from toybox.textutil import title_case


def test_title_case():
    assert title_case("hello world") == "Hello World"
    assert title_case("a") == "A"
