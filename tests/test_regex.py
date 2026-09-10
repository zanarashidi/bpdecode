import pytest

from bpdecode.regex import RegexSyntaxError, compile_regex


def accepts(pattern: str, text: str) -> bool:
    dfa = compile_regex(pattern)
    state = dfa.start
    for byte in text.encode("utf-8"):
        state = dfa.step(state, byte)
        if state == dfa.dead:
            return False
    return state in dfa.accept


@pytest.mark.parametrize(
    "pattern,text,ok",
    [
        ("abc", "abc", True),
        ("abc", "abd", False),
        ("a*", "", True),
        ("a*", "aaaa", True),
        ("a+", "", False),
        ("a?b", "b", True),
        ("a?b", "ab", True),
        ("a?b", "aab", False),
        ("foo|bar", "bar", True),
        ("foo|bar", "baz", False),
        ("(ab)+", "ababab", True),
        ("(ab)+", "aba", False),
        ("[0-9]+", "01234", True),
        ("[0-9]+", "12a", False),
        ("[^0-9]+", "abc", True),
        ("[^0-9]+", "ab1", False),
        (r"[0-9]{3}", "123", True),
        (r"[0-9]{3}", "12", False),
        (r"[0-9]{3}", "1234", False),
        (r"a{2,4}", "a", False),
        (r"a{2,4}", "aaa", True),
        (r"a{2,4}", "aaaaa", False),
        (r"x{2,}z", "xxxxz", True),
        (r"x{0,2}z", "z", True),
        (r"\d{4}-\d{2}", "2024-11", True),
        (r"a{4,2}", "", None),  # n < m -> syntax error
        (r"-?[0-9]+(\.[0-9]+)?", "-3.14", True),
        (r"-?[0-9]+(\.[0-9]+)?", "42", True),
        (r"-?[0-9]+(\.[0-9]+)?", "3.", False),
        (".", "x", True),
        ("a.c", "abc", True),
        # non-ASCII: patterns and text are lowered to UTF-8 bytes
        ("café", "café", True),
        ("caf.", "café", True),
        ("caf.", "cafX", True),
        ("[α-ω]+", "αβγδω", True),
        ("[α-ω]+", "αβΔ", False),
        ("🎂|🍰", "🍰", True),
        (".", "€", True),
    ],
)
def test_membership(pattern, text, ok):
    if ok is None:
        with pytest.raises(RegexSyntaxError):
            compile_regex(pattern)
        return
    assert accepts(pattern, text) is ok


def test_live_states_prunes_dead_paths():
    dfa = compile_regex("ab")
    live = dfa.live_states()
    # start and the state after 'a' are live; dead state is not
    assert dfa.start in live
    assert dfa.dead not in live


def test_unbalanced_paren():
    with pytest.raises(RegexSyntaxError):
        compile_regex("(ab")
