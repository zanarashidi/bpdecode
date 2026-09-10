"""RegexLogitsProcessor drives generation onto a grammar-valid path.

No model is loaded: a synthetic "model" emits fixed logits and we run the
generate() calling convention by hand.
"""

from __future__ import annotations

import re

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bpdecode.ops")

from bpdecode.hf import RegexLogitsProcessor  # noqa: E402
from bpdecode.reference import RegexConstraint  # noqa: E402
from bpdecode.tokenizer import Vocabulary  # noqa: E402

TOKENS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", "-", " ", "x", "</s>"]
VOCAB = Vocabulary.from_tokens(TOKENS, eos_id=len(TOKENS) - 1)


def _decode(ids: list[int]) -> str:
    return "".join(TOKENS[i] for i in ids if i != VOCAB.eos_id)


def _greedy(pattern: str, bias: torch.Tensor, max_new: int = 12) -> list[int]:
    """Argmax decode with the processor masking each step. `bias` is a fixed
    per-token logit added every step -- it stands in for the model's
    preferences and is chosen to fight the constraint.
    """
    lp = RegexLogitsProcessor(pattern, VOCAB)
    input_ids = torch.zeros(1, 1, dtype=torch.long)  # dummy prompt token
    out: list[int] = []
    for _ in range(max_new):
        scores = bias.clone().unsqueeze(0)
        scores = lp(input_ids, scores)
        nxt = int(scores[0].argmax())
        out.append(nxt)
        if nxt == VOCAB.eos_id:
            break
        input_ids = torch.cat([input_ids, torch.tensor([[nxt]])], dim=1)
    return out


@pytest.mark.parametrize(
    "pattern",
    ["-?[0-9]+(\\.[0-9]+)?", "[12][0-9][0-9]", "[0-9]+ [0-9]+"],
    ids=["decimal", "3digit", "two-ints"],
)
def test_every_step_stays_a_valid_prefix(pattern: str) -> None:
    # bias toward 'x' (id 13) every step -- no numeric pattern allows it, so the
    # mask has to override the model on every token.
    bias = torch.zeros(VOCAB.size)
    bias[TOKENS.index("x")] = 10.0
    ids = _greedy(pattern, bias)
    assert TOKENS.index("x") not in ids

    # the emitted string must be a prefix of some full match
    ref = RegexConstraint(pattern, VOCAB)
    for tok in ids:
        if tok == VOCAB.eos_id:
            assert ref.is_complete()
            break
        assert ref.accepts(tok), (pattern, _decode(ids), tok)
        ref.advance(tok)


def test_biasing_toward_eos_stops_only_when_complete() -> None:
    pattern = "[12][0-9][0-9]"  # exactly three digits
    bias = torch.zeros(VOCAB.size)
    bias[VOCAB.eos_id] = 10.0
    ids = _greedy(pattern, bias)
    assert _decode(ids) and re.fullmatch(pattern, _decode(ids)), ids
    assert ids[-1] == VOCAB.eos_id


def test_tracks_reference_step_by_step() -> None:
    pattern = "-?[0-9]+(\\.[0-9]+)?"
    lp = RegexLogitsProcessor(pattern, VOCAB)
    ref = RegexConstraint(pattern, VOCAB)
    input_ids = torch.zeros(1, 1, dtype=torch.long)

    for tok in [TOKENS.index("-"), TOKENS.index("3"), TOKENS.index("."), TOKENS.index("1")]:
        scores = torch.zeros(1, VOCAB.size)
        lp(input_ids, scores)
        allowed_lp = {i for i in range(VOCAB.size) if scores[0, i].item() == 0.0}
        allowed_ref = {i for i in range(VOCAB.size) if ref.accepts(i)}
        assert allowed_lp == allowed_ref
        ref.advance(tok)
        input_ids = torch.cat([input_ids, torch.tensor([[tok]])], dim=1)

    scores = torch.zeros(1, VOCAB.size)
    lp(input_ids, scores)
    assert lp.is_complete()
    assert scores[0, VOCAB.eos_id].item() == 0.0  # EOS now allowed


def test_reset_allows_reuse() -> None:
    lp = RegexLogitsProcessor("[0-9]+", VOCAB)
    ids1 = _greedy_with(lp, torch.zeros(VOCAB.size))
    lp.reset()
    ids2 = _greedy_with(lp, torch.zeros(VOCAB.size))
    assert ids1 == ids2


def _greedy_with(lp: RegexLogitsProcessor, bias: torch.Tensor) -> list[int]:
    input_ids = torch.zeros(1, 1, dtype=torch.long)
    out: list[int] = []
    for _ in range(6):
        scores = lp(input_ids, bias.clone().unsqueeze(0))
        nxt = int(scores[0].argmax())
        out.append(nxt)
        if nxt == VOCAB.eos_id:
            break
        input_ids = torch.cat([input_ids, torch.tensor([[nxt]])], dim=1)
    return out


def test_half_precision_logits_are_handled() -> None:
    lp = RegexLogitsProcessor("[0-9]+", VOCAB)
    scores = torch.zeros(1, VOCAB.size, dtype=torch.float16)
    scores[0, TOKENS.index("x")] = 5.0
    out = lp(torch.zeros(1, 1, dtype=torch.long), scores)
    assert out.dtype == torch.float16
    assert torch.isneginf(out[0, TOKENS.index("x")])
    assert out[0, TOKENS.index("0")].item() == 0.0
