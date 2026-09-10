"""The dense tok_next table and its gather fast path must match the scalar
byte-walk (`step`) and the kernel ops token for token.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
ops = pytest.importorskip("bpdecode.ops")

from bpdecode.fsa import fsa_from_dfa, step, token_symbols  # noqa: E402
from bpdecode.ops import FsaTensors, build_token_transitions  # noqa: E402
from bpdecode.regex import compile_regex  # noqa: E402
from bpdecode.tokenizer import Vocabulary  # noqa: E402

VOCAB = Vocabulary.from_tokens(
    ["a", "b", "c", "ab", "ba", "0", "1", "12", "-", ".", "aa", "cafe", "\xc3", "<eos>"],
    eos_id=13,
)
PATTERNS = [
    "[01]+",
    "a+b*",
    "(ab)+",
    "-?[0-9]+(\\.[0-9]+)?",
    "a|bc|12",
    "[a-c]*",
    "caf(e|)",
]


def _states(dfa):
    seen, stack = {dfa.start}, [dfa.start]
    while stack:
        for t in dfa.trans[stack.pop()]:
            if t not in seen:
                seen.add(t)
                stack.append(t)
    return sorted(seen)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_tok_next_matches_step(pattern: str) -> None:
    dfa = compile_regex(pattern)
    fsa_t = fsa_from_dfa(dfa)
    toks = token_symbols(dfa, VOCAB)
    dense = build_token_transitions(FsaTensors.build(pattern, VOCAB, dense=False))

    for s in range(fsa_t.num_states):
        for t in range(VOCAB.size):
            assert int(dense[s, t]) == step(fsa_t, toks, s, t), (pattern, s, t)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_dense_apply_mask_matches_kernel(pattern: str) -> None:
    dfa = compile_regex(pattern)
    states = _states(dfa)
    st = torch.tensor(states, dtype=torch.int32)
    plain = FsaTensors.build(pattern, VOCAB, dense=False)
    dense = plain.densify()

    a = torch.randn(len(states), VOCAB.size)
    want, got = a.clone(), a.clone()
    ops.apply_mask_(want, plain, st)  # kernel path
    ops.apply_mask_(got, dense, st)  # gather path
    assert torch.equal(want, got)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_dense_advance_state_matches_kernel(pattern: str) -> None:
    dfa = compile_regex(pattern)
    states, tokens = [], []
    for s in _states(dfa):
        for t in range(VOCAB.size):
            states.append(s)
            tokens.append(t)
    st = torch.tensor(states, dtype=torch.int32)
    tk = torch.tensor(tokens, dtype=torch.int32)
    plain = FsaTensors.build(pattern, VOCAB, dense=False)

    assert torch.equal(
        ops.advance_state(plain, st, tk), ops.advance_state(plain.densify(), st, tk)
    )


def test_dense_handles_broken_state() -> None:
    plain = FsaTensors.build("[01]+", VOCAB, dense=False).densify()
    st = torch.tensor([-1, 0], dtype=torch.int32)
    tk = torch.tensor([VOCAB.token_bytes.index(b"0")] * 2, dtype=torch.int32)
    out = ops.advance_state(plain, st, tk)
    assert int(out[0]) == -1  # broken stays broken


def test_auto_dense_threshold() -> None:
    small = FsaTensors.build("[01]+", VOCAB)  # tiny -> auto densifies
    assert small.tok_next is not None
    off = FsaTensors.build("[01]+", VOCAB, dense=False)
    assert off.tok_next is None
