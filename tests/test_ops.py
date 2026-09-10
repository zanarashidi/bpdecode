"""torch.ops.bpdecode.* on CPU tensors must match the scalar reference.

The CUDA path shares the kernel source with these entry points; it is exercised
on GPU CI (see .github/workflows/gpu.yml), not here.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
ops = pytest.importorskip("bpdecode.ops")

from bpdecode.automaton import DEAD, TokenDFA  # noqa: E402
from bpdecode.fsa import apply_mask, compute_mask, fsa_from_dfa, token_symbols  # noqa: E402
from bpdecode.reference import RegexConstraint  # noqa: E402
from bpdecode.regex import compile_regex  # noqa: E402
from bpdecode.tokenizer import Vocabulary  # noqa: E402

FsaTensors = ops.FsaTensors

VOCAB = Vocabulary.from_tokens(
    ["a", "b", "c", "ab", "ba", "0", "1", "12", "-", ".", "aa", "", "<eos>"],
    eos_id=12,
)
PATTERNS = ["[01]+", "a+b*", "(ab)+", "-?[0-9]+(\\.[0-9]+)?", "a|bc|12", "[a-c]*"]


def _states(dfa) -> list[int]:
    seen, stack = {dfa.start}, [dfa.start]
    while stack:
        for t in dfa.trans[stack.pop()]:
            if t not in seen:
                seen.add(t)
                stack.append(t)
    return sorted(seen)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_apply_mask_matches_scalar(pattern: str) -> None:
    dfa = compile_regex(pattern)
    fsa = fsa_from_dfa(dfa)
    toks = token_symbols(dfa, VOCAB)
    bundle = FsaTensors.build(pattern, VOCAB)

    states = _states(dfa)
    logits = torch.zeros(len(states), VOCAB.size)
    ops.apply_mask_(logits, bundle, torch.tensor(states, dtype=torch.int32))

    ref = apply_mask(fsa, toks, states, [[0.0] * VOCAB.size for _ in states])
    assert logits.tolist() == ref


@pytest.mark.parametrize("pattern", PATTERNS)
def test_advance_state_matches_tokendfa(pattern: str) -> None:
    dfa = compile_regex(pattern)
    tdfa = TokenDFA(dfa, VOCAB)
    bundle = FsaTensors.build(pattern, VOCAB)

    states, tokens, expected = [], [], []
    for s in _states(dfa):
        for tid in range(VOCAB.size):
            states.append(s)
            tokens.append(tid)
            ref = tdfa.step(s, tid)
            expected.append(-1 if ref == DEAD else ref)

    got = ops.advance_state(
        bundle,
        torch.tensor(states, dtype=torch.int32),
        torch.tensor(tokens, dtype=torch.int32),
    )
    assert got.tolist() == expected


def test_build_reachability_op_matches_export() -> None:
    dfa = compile_regex("-?[0-9]+(\\.[0-9]+)?")
    bundle = FsaTensors.build(dfa, VOCAB)
    live = torch.ops.bpdecode.build_reachability(
        bundle.trans, bundle.accept, bundle.num_symbols
    )
    assert live.tolist() == bundle.live.tolist()


def test_apply_mask_agrees_with_regexconstraint() -> None:
    pattern = "[01]+"
    con = RegexConstraint(pattern, VOCAB)
    con.advance(VOCAB.token_bytes.index(b"0"))
    bundle = FsaTensors.build(pattern, VOCAB)

    logits = torch.zeros(1, VOCAB.size)
    ops.apply_mask_(logits, bundle, torch.tensor([con.state], dtype=torch.int32))
    assert logits[0].tolist() == con.apply_([0.0] * VOCAB.size)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_compute_mask_op_bits_match_scalar(pattern: str) -> None:
    dfa = compile_regex(pattern)
    fsa = fsa_from_dfa(dfa)
    toks = token_symbols(dfa, VOCAB)
    bundle = FsaTensors.build(pattern, VOCAB)
    states = _states(dfa)

    packed = ops.compute_mask(bundle, torch.tensor(states, dtype=torch.int32))
    for row, s in zip(packed.tolist(), states, strict=True):
        scalar = compute_mask(fsa, toks, s)
        for t in range(VOCAB.size):
            bit = (row[t >> 5] >> (t & 31)) & 1
            assert bool(bit) == scalar[t], (pattern, s, t)
