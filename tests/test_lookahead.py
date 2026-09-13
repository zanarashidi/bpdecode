"""Unit tests for bpdecode.lookahead's model-free scoring arithmetic.

The rest of bpdecode.lookahead.generate_model_weighted needs a real HF model
and its KV cache (see bench/lookahead_model_weighted.py, run manually --
this repo's convention is real-model checks live in bench/, not tests/).
_score_candidates is pulled out specifically so the one place a plain bug
(NaN, wrong sign) could hide is testable without one.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bpdecode.lookahead import _score_candidates  # noqa: E402

NEG_INF = float("-inf")


def test_matches_plain_formula_when_no_dead_candidates() -> None:
    cand_logits = torch.tensor([1.0, 2.0, -3.0])
    weight = torch.tensor([-0.5, -2.0, -0.1])
    for alpha in (0.0, 0.5, 1.0, 3.0):
        got = _score_candidates(cand_logits, weight, alpha)
        want = cand_logits + alpha * weight
        assert torch.allclose(got, want)


def test_dead_candidate_scores_neg_inf_at_every_alpha() -> None:
    cand_logits = torch.tensor([1.0, 2.0])
    weight = torch.tensor([NEG_INF, -0.5])
    for alpha in (0.0, 1.0, 5.0):
        got = _score_candidates(cand_logits, weight, alpha)
        assert got[0].item() == NEG_INF
        assert not torch.isnan(got[0])


def test_alpha_zero_no_nan_even_with_all_dead() -> None:
    # the bug this guards: alpha * -inf is nan at alpha == 0, not -inf.
    cand_logits = torch.tensor([1.0, -5.0, 0.0])
    weight = torch.full((3,), NEG_INF)
    got = _score_candidates(cand_logits, weight, 0.0)
    assert not torch.isnan(got).any()
    assert torch.isneginf(got).all()


def test_already_disallowed_candidate_stays_neg_inf() -> None:
    # a candidate whose own logit was -inf (spurious top-k pick, fewer than
    # k tokens were actually allowed) combined with a dead weight must not
    # produce nan or a finite score via -inf + -inf special-casing.
    cand_logits = torch.tensor([NEG_INF])
    weight = torch.tensor([NEG_INF])
    got = _score_candidates(cand_logits, weight, 1.0)
    assert got.item() == NEG_INF


def test_higher_alpha_prefers_higher_weight_among_viable() -> None:
    cand_logits = torch.tensor([0.0, 0.0])
    weight = torch.tensor([-2.0, -0.1])  # second candidate is more likely to stay valid
    score = _score_candidates(cand_logits, weight, alpha=2.0)
    assert score.argmax().item() == 1
