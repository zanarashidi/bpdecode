"""ConstraintBatch must track per-sequence grammar state exactly like the
single-sequence reference, through a continuous-batching lifecycle.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bpdecode.ops")

from bpdecode.batch import (  # noqa: E402
    BROKEN,
    FREE,
    ConstraintBatch,
    GrammarCache,
    MaskCache,
)
from bpdecode.reference import RegexConstraint  # noqa: E402
from bpdecode.tokenizer import Vocabulary  # noqa: E402

TOKENS = ["a", "b", "c", "ab", "0", "1", "12", "-", ".", "<eos>"]
VOCAB = Vocabulary.from_tokens(TOKENS, eos_id=9)


def test_grammar_cache_dedups_and_evicts_lru() -> None:
    cache = GrammarCache(VOCAB, max_size=2)
    a1 = cache.get("[01]+")
    a2 = cache.get("[01]+")
    assert a1 is a2  # identical grammar -> shared FsaTensors
    cache.get("a+")
    cache.get("[01]+")  # touch -> most recent
    cache.get("b+")  # evicts "a+", not "[01]+"
    assert len(cache) == 2
    assert cache.get("[01]+") is a1


def test_lifecycle_slots_are_reused() -> None:
    cache = GrammarCache(VOCAB)
    batch = ConstraintBatch(cache.get("[01]+"), capacity=2)
    s0 = batch.add("r0")
    s1 = batch.add("r1")
    assert {s0, s1} == {0, 1}
    with pytest.raises(RuntimeError):
        batch.add("r2")  # full
    batch.evict("r0")
    assert batch._state[s0] == FREE
    assert batch.add("r2") == s0  # freed slot reused
    assert "r0" not in batch and "r2" in batch


def _simulate(pattern: str, token_seq: list[int], mask_cache: bool = False) -> None:
    """Drive one sequence through ConstraintBatch and RegexConstraint in
    lockstep, asserting the allow-sets match at every step.
    """
    cache = GrammarCache(VOCAB)
    batch = ConstraintBatch(cache.get(pattern), capacity=4, mask_cache=mask_cache)
    ref = RegexConstraint(pattern, VOCAB)
    batch.add("s")

    for tok in token_seq:
        logits = torch.zeros(1, VOCAB.size)
        batch.apply_mask(["s"], logits)
        allowed_batch = {i for i in range(VOCAB.size) if logits[0, i].item() == 0.0}
        allowed_ref = {i for i in range(VOCAB.size) if ref.accepts(i)}
        assert allowed_batch == allowed_ref, (pattern, tok)

        if tok not in allowed_ref:
            break
        batch.commit(["s"], torch.tensor([tok], dtype=torch.int32))
        ref.advance(tok)
        assert batch.is_complete("s") == ref.is_complete()


@pytest.mark.parametrize("mask_cache", [False, True], ids=["kernel", "cached"])
@pytest.mark.parametrize(
    "pattern,seq",
    [
        ("[01]+", [4, 5, 4, 9]),  # 0 1 0 <eos>
        ("-?[0-9]+(\\.[0-9]+)?", [7, 4, 8, 5]),  # - 0 . 1
        ("(ab)+", [3, 3, 9]),  # ab ab <eos>
        ("a+b*", [0, 0, 1, 9]),
    ],
)
def test_matches_single_sequence_reference(
    pattern: str, seq: list[int], mask_cache: bool
) -> None:
    _simulate(pattern, seq, mask_cache=mask_cache)


def test_batch_rows_are_independent() -> None:
    cache = GrammarCache(VOCAB)
    batch = ConstraintBatch(cache.get("[01]+"), capacity=3)
    for r in ("a", "b", "c"):
        batch.add(r)

    # advance each row by a different amount
    batch.commit(["a", "b", "c"], torch.tensor([4, 5, 4], dtype=torch.int32))
    batch.commit(["b"], torch.tensor([5], dtype=torch.int32))

    logits = torch.zeros(3, VOCAB.size)
    batch.apply_mask(["a", "b", "c"], logits)
    # all three are mid-number: 0,1 allowed, <eos> allowed (accepting), letters not
    for row in range(3):
        assert logits[row, TOKENS.index("0")].item() == 0.0
        assert logits[row, TOKENS.index("a")].item() == float("-inf")
        assert logits[row, VOCAB.eos_id].item() == 0.0


@pytest.mark.parametrize(
    "pattern", ["[01]+", "-?[0-9]+(\\.[0-9]+)?", "(ab)+", "a|b|12"]
)
def test_mask_cache_matches_uncached(pattern: str) -> None:
    from bpdecode.ops import apply_mask_
    from bpdecode.regex import compile_regex

    fsa = GrammarCache(VOCAB).get(pattern)
    dfa = compile_regex(pattern)
    states = sorted(range(len(dfa.trans)))  # every state incl. dead
    st = torch.tensor(states * 2, dtype=torch.int32)  # repeats -> exercise grouping

    mc = MaskCache(fsa)
    base = torch.randn(len(st), VOCAB.size)
    want = base.clone()
    apply_mask_(want, fsa, st)
    got = base.clone()
    mc.apply(st, got)
    assert torch.equal(want, got)

    # a second pass is all cache hits and identical
    got2 = base.clone()
    mc.apply(st, got2)
    assert torch.equal(want, got2)
    assert len(mc) == len(set(states))


def test_mask_cache_lru_evicts() -> None:
    fsa = GrammarCache(VOCAB).get("-?[0-9]+(\\.[0-9]+)?")
    mc = MaskCache(fsa, max_states=2)
    mc.allowed_ids(0)
    mc.allowed_ids(1)
    mc.allowed_ids(0)  # touch
    mc.allowed_ids(2)  # evicts state 1
    assert len(mc) == 2
    assert 1 not in mc._allowed and 0 in mc._allowed


def test_broken_state_is_flagged() -> None:
    cache = GrammarCache(VOCAB)
    batch = ConstraintBatch(cache.get("[01]+"), capacity=1)
    batch.add("s")
    batch.commit(["s"], torch.tensor([TOKENS.index("a")], dtype=torch.int32))
    assert batch.state_of("s") == BROKEN
    assert batch.is_broken("s")
