"""Reference-constraint behaviour, plus a brute-force differential check.

The brute-force oracle enumerates every token string up to a bounded length,
keeps the ones the DFA accepts, and checks that :class:`RegexConstraint`
allows a token at step ``k`` iff some accepted string has that token at ``k``.
"""

import itertools

import pytest

from bpdecode import RegexConstraint, Vocabulary
from bpdecode.automaton import TokenDFA
from bpdecode.regex import compile_regex

TOKENS = ["a", "b", "ab", "c", "0", "1", ".", "-", "<eos>"]
EOS = len(TOKENS) - 1
VOCAB = Vocabulary.from_tokens(TOKENS, eos_id=EOS)


def brute_force_valid_strings(pattern: str, max_tokens: int) -> list[tuple[int, ...]]:
    dfa = compile_regex(pattern)
    ids = [i for i in range(len(TOKENS)) if i != EOS]
    valid = []
    for length in range(max_tokens + 1):
        for combo in itertools.product(ids, repeat=length):
            text = "".join(TOKENS[i] for i in combo)
            state = dfa.start
            dead = False
            for ch in text:
                state = dfa.step(state, ord(ch))
                if state == dfa.dead:
                    dead = True
                    break
            if not dead and state in dfa.accept:
                valid.append(combo)
    return valid


@pytest.mark.parametrize("pattern", ["ab", "a+", "[01]+", "(ab|c)+", "-?[01]+"])
def test_matches_brute_force(pattern):
    # Enumerate to depth MAX_TOKENS but only assert for prefixes shallow enough
    # that the remaining token budget can realise any legitimate next token.
    max_tokens = 6
    probe_depth = 3
    valid = brute_force_valid_strings(pattern, max_tokens=max_tokens)
    assert valid, "test pattern should have some short solutions"

    for prefix_len in range(probe_depth):
        prefixes = {v[:prefix_len] for v in valid if len(v) >= prefix_len}
        for prefix in prefixes:
            con = RegexConstraint(pattern, VOCAB)
            for tid in prefix:
                con.advance(tid)
            expected_next = {
                v[prefix_len]
                for v in valid
                if len(v) > prefix_len and v[:prefix_len] == prefix
            }
            if any(v == prefix for v in valid):
                expected_next.add(EOS)
            got = {t for t in range(len(TOKENS)) if con.accepts(t)}
            assert got == expected_next, (pattern, prefix)


def test_eos_only_when_accepting():
    con = RegexConstraint("ab", VOCAB)
    assert not con.accepts(EOS)
    con.advance(TOKENS.index("ab"))
    assert con.accepts(EOS)
    assert con.is_complete()


def test_advance_rejects_invalid_token():
    con = RegexConstraint("ab", VOCAB)
    with pytest.raises(ValueError):
        con.advance(TOKENS.index("c"))


def test_apply_masks_logits():
    con = RegexConstraint("[01]+", VOCAB)
    scores = [0.0] * len(TOKENS)
    con.apply_(scores)
    assert scores[TOKENS.index("0")] == 0.0
    assert scores[TOKENS.index("1")] == 0.0
    assert scores[TOKENS.index("a")] == float("-inf")


def test_from_token_dfa_shares_cache():
    tdfa = TokenDFA(compile_regex("a+"), VOCAB)
    c1 = RegexConstraint.from_token_dfa(tdfa)
    c2 = RegexConstraint.from_token_dfa(tdfa)
    c1.advance(TOKENS.index("a"))
    assert c2.state == tdfa.start  # independent state
    assert c1._tdfa is c2._tdfa   # shared automaton/cache
