"""The exported FSA tables must reproduce the TokenDFA reference exactly.

`bpdecode.fsa` flattens a compiled DFA + vocabulary into the POD arrays the
C++/CUDA core consumes, and re-implements the scalar `step` / `compute_mask`
semantics.  These tests pin that export against `TokenDFA`, the Phase 0 oracle.
"""

from __future__ import annotations

import pytest

from bpdecode.automaton import DEAD, TokenDFA
from bpdecode.fsa import (
    build_reachability,
    compute_mask,
    fsa_from_dfa,
    step,
    token_symbols,
)
from bpdecode.regex import compile_regex
from bpdecode.tokenizer import Vocabulary

VOCAB = Vocabulary.from_tokens(
    ["a", "b", "c", "ab", "ba", "0", "1", "12", "-", ".", "aa", "", "<eos>"],
    eos_id=12,
)

PATTERNS = [
    "[01]+",
    "a+b*",
    "(ab)+",
    "-?[0-9]+(\\.[0-9]+)?",
    "a|bc|12",
    "[a-c]*",
    "aa+",
    "c",
]


def _reachable_states(dfa) -> set[int]:
    seen = {dfa.start}
    stack = [dfa.start]
    while stack:
        s = stack.pop()
        for t in dfa.trans[s]:
            if t not in seen:
                seen.add(t)
                stack.append(t)
    return seen


@pytest.mark.parametrize("pattern", PATTERNS)
def test_reachability_matches_dfa_live_states(pattern: str) -> None:
    dfa = compile_regex(pattern)
    fsa = fsa_from_dfa(dfa)
    live = build_reachability(fsa.num_states, fsa.num_symbols, fsa.trans, fsa.accept)
    assert live == list(fsa.live)
    expected = dfa.live_states()
    for s in range(fsa.num_states):
        assert bool(live[s]) == (s in expected)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_step_and_mask_match_tokendfa(pattern: str) -> None:
    dfa = compile_regex(pattern)
    tdfa = TokenDFA(dfa, VOCAB)
    fsa = fsa_from_dfa(dfa)
    toks = token_symbols(dfa, VOCAB)

    for state in _reachable_states(dfa):
        ref_mask = tdfa.mask(state)
        got_mask = compute_mask(fsa, toks, state)
        assert got_mask == ref_mask, (pattern, state)

        for tid in range(VOCAB.size):
            ref = tdfa.step(state, tid)
            got = step(fsa, toks, state, tid)
            # TokenDFA returns the concrete next state; fsa.step agrees on it
            # except that DEAD is the only "rejected" sentinel in both.
            if ref == DEAD:
                assert got == -1, (pattern, state, tid)
            else:
                assert got == ref, (pattern, state, tid)


def test_multibyte_tokens_split_across_the_utf8_automaton() -> None:
    # byte-level BPE routinely splits one code point across several tokens;
    # the export must walk those partial byte sequences correctly.
    # tokens: 0=c 1=a 2=f 3=0xC3 4=0xA9 5=0xC3A9 6=x 7=eos
    vocab = Vocabulary.from_tokens(
        [b"c", b"a", b"f", b"\xc3", b"\xa9", b"\xc3\xa9", b"x", b"<eos>"],
        eos_id=7,
    )
    dfa = compile_regex("caf(é|e)")
    tdfa = TokenDFA(dfa, vocab)
    fsa = fsa_from_dfa(dfa)
    toks = token_symbols(dfa, vocab)

    for state in _reachable_states(dfa):
        assert compute_mask(fsa, toks, state) == tdfa.mask(state), state

    # "caf" then the lone 0xC3 lead byte is a live partial-UTF-8 prefix of "café"
    s = tdfa.start
    for tid in (0, 1, 2, 3):
        s = tdfa.step(s, tid)
        assert s != DEAD
    assert tdfa.step(s, 4) != DEAD  # 0xA9 completes é -> accepting
    assert tdfa.step(s, 5) == DEAD  # a second 0xC3A9 pair is not valid here
    assert step(fsa, toks, s, 4) == tdfa.step(s, 4)


def test_eos_only_at_accepting_states() -> None:
    dfa = compile_regex("[01]+")
    tdfa = TokenDFA(dfa, VOCAB)
    fsa = fsa_from_dfa(dfa)
    toks = token_symbols(dfa, VOCAB)
    for state in _reachable_states(dfa):
        allowed_eos = compute_mask(fsa, toks, state)[VOCAB.eos_id]
        assert allowed_eos == tdfa.is_accepting(state)
