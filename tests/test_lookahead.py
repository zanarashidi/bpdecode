"""Unit tests for bpdecode.lookahead's model-free pieces: the scoring
arithmetic (_score_candidates) and the grammar-state adapters (_regex_ops,
_cfg_ops, _select_winner) that _generate_loop drives -- all of it operates
on plain state tensors and doesn't touch the model, so it's testable without
one. The full loop (bench/lookahead_model_weighted.py) needs a real HF model
and its KV cache; this repo's convention is real-model checks live in
bench/, not tests/.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bpdecode.grammar.constraint import CFGConstraint  # noqa: E402
from bpdecode.lookahead import (  # noqa: E402
    _cfg_ops,
    _regex_ops,
    _score_candidates,
    _select_winner,
)
from bpdecode.tokenizer import Vocabulary  # noqa: E402

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


VOCAB = Vocabulary.from_tokens(
    ["(", ")", "x", "y", "{", "}", "a", "b", ":", "0", "1", ",", "<eos>"], eos_id=12
)


def test_select_winner_handles_1d_and_2d_state() -> None:
    # 1D (regex-style) state: [n * kk] -> pick one of kk per row -> [n]
    x1 = torch.arange(6)  # n=2, kk=3: rows [0,1,2] and [3,4,5]
    got = _select_winner(x1, n=2, kk=3, best=torch.tensor([2, 0]))
    assert got.tolist() == [2, 3]

    # 2D (CFG-style) state: [n * kk, F] -> pick one of kk per row -> [n, F]
    x2 = torch.arange(6 * 4).view(6, 4)
    got2 = _select_winner(x2, n=2, kk=3, best=torch.tensor([2, 0]))
    assert torch.equal(got2[0], x2[2])
    assert torch.equal(got2[1], x2[3])


def test_regex_ops_mask_and_advance_match_reference() -> None:
    ops = _regex_ops(r"[01]+", VOCAB, torch.device("cpu"))
    n = 3
    states = ops.init(n)
    logits = torch.zeros(n, VOCAB.size)
    ops.mask(logits, states)
    allowed_per_row = [
        {i for i in range(VOCAB.size) if logits[r, i].item() == 0.0} for r in range(n)
    ]
    assert all(a == allowed_per_row[0] for a in allowed_per_row)
    one = VOCAB.token_bytes.index(b"1")
    nxt = ops.advance(states, torch.full((n,), one, dtype=torch.long))
    assert (nxt >= 0).all()


def test_cfg_ops_mask_and_advance_match_cfgconstraint() -> None:
    src = 'root ::= "(" root ")" | "x" | "y"'
    ops = _cfg_ops(src, VOCAB, torch.device("cpu"), root="root")
    ref = CFGConstraint(src, VOCAB)

    states = ops.init(1)
    logits = torch.zeros(1, VOCAB.size)
    ops.mask(logits, states)
    got = {i for i in range(VOCAB.size) if logits[0, i].item() == 0.0}
    want = {i for i in range(VOCAB.size) if ref.accepts(i)}
    assert got == want

    paren = VOCAB.token_bytes.index(b"(")
    states = ops.advance(states, torch.tensor([paren], dtype=torch.long))
    ref.advance(paren)
    logits2 = torch.zeros(1, VOCAB.size)
    ops.mask(logits2, states)
    got2 = {i for i in range(VOCAB.size) if logits2[0, i].item() == 0.0}
    want2 = {i for i in range(VOCAB.size) if ref.accepts(i)}
    assert got2 == want2


def test_cfg_ops_branch_repeat_and_select_round_trips() -> None:
    # exercises exactly what _generate_loop does to a 2D (CFG) state tensor:
    # repeat_interleave(dim=0) for kk branches, advance each independently,
    # then _select_winner picks one branch back down to n rows.
    src = 'root ::= "(" root ")" | "x" | "y"'
    ops = _cfg_ops(src, VOCAB, torch.device("cpu"), root="root")
    n, kk = 2, 3
    states = ops.init(n)
    branch = states.repeat_interleave(kk, dim=0)
    assert branch.shape[0] == n * kk

    paren, x = VOCAB.token_bytes.index(b"("), VOCAB.token_bytes.index(b"x")
    tokens = torch.tensor([paren, x, x, x, paren, paren], dtype=torch.long)
    branch = ops.advance(branch, tokens)

    best = torch.tensor([0, 2])  # row 0 takes '(', row 1 takes '('
    winner = _select_winner(branch, n, kk, best)
    assert winner.shape == states.shape

    ref0, ref1 = CFGConstraint(src, VOCAB), CFGConstraint(src, VOCAB)
    ref0.advance(paren)
    ref1.advance(paren)
    logits = torch.zeros(n, VOCAB.size)
    ops.mask(logits, winner)
    for r, ref in ((0, ref0), (1, ref1)):
        got = {i for i in range(VOCAB.size) if logits[r, i].item() == 0.0}
        want = {i for i in range(VOCAB.size) if ref.accepts(i)}
        assert got == want
