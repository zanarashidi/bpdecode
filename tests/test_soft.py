"""Soft lookahead: the k-step backward sum-product bias table."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
ops = pytest.importorskip("bpdecode.ops")

from bpdecode.ops import FsaTensors, apply_mask_, apply_soft_  # noqa: E402
from bpdecode.tokenizer import Vocabulary  # noqa: E402

VOCAB = Vocabulary.from_tokens(
    ["a", "b", "c", "ab", "ba", "0", "1", "12", "-", ".", "ge", "<eos>"], eos_id=11
)


def _states(dfa):
    seen, stack = {dfa.start}, [dfa.start]
    while stack:
        for t in dfa.trans[stack.pop()]:
            if t not in seen:
                seen.add(t)
                stack.append(t)
    return sorted(seen)


@pytest.mark.parametrize("pattern", ["[01]+", "a+b*", "(ab)+", "-?[0-9]+(\\.[0-9]+)?"])
def test_lookahead_masks_exactly_like_hard_mask(pattern: str) -> None:
    fsa = FsaTensors.build(pattern, VOCAB, dense=False).with_lookahead(3)
    from bpdecode.regex import compile_regex

    states = torch.tensor(_states(compile_regex(pattern)), dtype=torch.int32)

    la = fsa.lookahead.index_select(0, states.long())
    want_allowed = torch.zeros(len(states), VOCAB.size)
    apply_mask_(want_allowed, fsa, states)  # 0 where allowed, -inf where not
    assert torch.equal(torch.isneginf(la), torch.isneginf(want_allowed))


@pytest.mark.parametrize("pattern", ["[01]+", "a+b*", "(ab)+"])
def test_alpha_zero_is_hard_masking(pattern: str) -> None:
    fsa = FsaTensors.build(pattern, VOCAB, dense=False).with_lookahead(3)
    from bpdecode.regex import compile_regex

    states = torch.tensor(_states(compile_regex(pattern)), dtype=torch.int32)
    base = torch.randn(len(states), VOCAB.size)

    hard = base.clone()
    apply_mask_(hard, fsa, states)
    soft0 = base.clone()
    apply_soft_(soft0, fsa, states, alpha=0.0)
    assert torch.equal(hard, soft0)


def test_more_open_continuations_score_higher() -> None:
    # from start: "a" forces exactly the string "abbbb..."? no -- design a clear case
    #   root: "a" then (any of 3 digits)+   vs   "b" then exactly "0"
    # after "a" many length-2 futures; after "b" only "b0"/"b0<eos>"-ish
    fsa = FsaTensors.build("a[012]+|b0", VOCAB, dense=False).with_lookahead(3)
    la_start = fsa.lookahead[fsa.start]
    a = la_start[VOCAB.token_bytes.index(b"a")].item()
    b = la_start[VOCAB.token_bytes.index(b"b")].item()
    assert math.isfinite(a) and math.isfinite(b)
    assert a > b  # "a" keeps more of the language open


def test_dead_end_token_becomes_neg_inf() -> None:
    # only completion of "a" is "ge" (one token). After that, done.
    # "a" alone is byte-valid but Z_1(state after "a") counts just {emit "ge"}.
    fsa = FsaTensors.build("age", VOCAB, dense=False).with_lookahead(3)
    la = fsa.lookahead[fsa.start]
    # token "a" is allowed (prefix of "age"); "ge" completes it
    assert not torch.isneginf(la[VOCAB.token_bytes.index(b"a")])
    # but a token with no valid continuation at all is -inf
    assert torch.isneginf(la[VOCAB.token_bytes.index(b"b")])


def test_apply_soft_adds_scaled_bias() -> None:
    fsa = FsaTensors.build("[012]+", VOCAB, dense=False).with_lookahead(3)
    states = torch.tensor([fsa.start], dtype=torch.int32)
    logits = torch.zeros(1, VOCAB.size)
    apply_soft_(logits, fsa, states, alpha=2.0)
    expected = 2.0 * fsa.lookahead[fsa.start]
    finite = torch.isfinite(expected)
    assert torch.allclose(logits[0][finite], expected[finite])
    assert torch.isneginf(logits[0][~finite]).all()


def test_k1_lookahead_is_flat_over_allowed() -> None:
    fsa = FsaTensors.build("[01]+", VOCAB, dense=False).with_lookahead(1)
    la = fsa.lookahead[fsa.start]
    allowed = la[torch.isfinite(la)]
    assert torch.allclose(allowed, torch.zeros_like(allowed))  # logZ_0 == 0
