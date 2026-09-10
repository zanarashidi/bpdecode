"""ConstraintBatch must track per-sequence grammar state exactly like the
single-sequence reference, through a continuous-batching lifecycle.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bpdecode.ops")

from bpdecode.batch import BROKEN, FREE, ConstraintBatch, GrammarCache  # noqa: E402
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


def _simulate(pattern: str, token_seq: list[int]) -> None:
    """Drive one sequence through ConstraintBatch and RegexConstraint in
    lockstep, asserting the allow-sets match at every step.
    """
    cache = GrammarCache(VOCAB)
    batch = ConstraintBatch(cache.get(pattern), capacity=4)
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


@pytest.mark.parametrize(
    "pattern,seq",
    [
        ("[01]+", [4, 5, 4, 9]),  # 0 1 0 <eos>
        ("-?[0-9]+(\\.[0-9]+)?", [7, 4, 8, 5]),  # - 0 . 1
        ("(ab)+", [3, 3, 9]),  # ab ab <eos>
        ("a+b*", [0, 0, 1, 9]),
    ],
)
def test_matches_single_sequence_reference(pattern: str, seq: list[int]) -> None:
    _simulate(pattern, seq)


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


def test_broken_state_is_flagged() -> None:
    cache = GrammarCache(VOCAB)
    batch = ConstraintBatch(cache.get("[01]+"), capacity=1)
    batch.add("s")
    batch.commit(["s"], torch.tensor([TOKENS.index("a")], dtype=torch.int32))
    assert batch.state_of("s") == BROKEN
    assert batch.is_broken("s")
