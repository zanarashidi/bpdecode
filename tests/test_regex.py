import pytest

from bpdecode.regex import RegexSyntaxError, compile_regex


def accepts(pattern: str, text: str) -> bool:
    dfa = compile_regex(pattern)
    state = dfa.start
    for ch in text:
        state = dfa.step(state, ord(ch))
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
        (r"\d{0,0}", "", None),  # counted repetition -> syntax error
        (r"-?[0-9]+(\.[0-9]+)?", "-3.14", True),
        (r"-?[0-9]+(\.[0-9]+)?", "42", True),
        (r"-?[0-9]+(\.[0-9]+)?", "3.", False),
        (".", "x", True),
        ("a.c", "abc", True),
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
