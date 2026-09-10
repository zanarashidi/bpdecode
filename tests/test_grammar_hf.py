"""GrammarLogitsProcessor drives generation onto a CFG-valid path (no model)."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from bpdecode.grammar.constraint import CFGConstraint  # noqa: E402
from bpdecode.hf import GrammarLogitsProcessor  # noqa: E402
from bpdecode.tokenizer import Vocabulary  # noqa: E402

TOKENS = [
    "{", "}", "[", "]", '"', ":", ",", " ",
    "a", "b", "c", "0", "1", "2", "true", "false", "null",
    "name", "age", "</s>",
]
VOCAB = Vocabulary.from_tokens(TOKENS, eos_id=len(TOKENS) - 1)


def _greedy(lp: GrammarLogitsProcessor, bias: torch.Tensor, max_new: int = 40) -> str:
    """Argmax decode; ties broken toward the longer token, the way a real model
    that "knows the word" would (hard-masking alone can't escape partial-token
    dead ends).
    """
    input_ids = torch.zeros(1, 1, dtype=torch.long)
    out: list[int] = []
    for _ in range(max_new):
        scores = lp(input_ids, bias.clone().unsqueeze(0))[0]
        best = max(
            (i for i in range(VOCAB.size) if scores[i].item() > -1e30),
            key=lambda i: (scores[i].item(), len(TOKENS[i])),
        )
        if best == VOCAB.eos_id:
            break
        out.append(best)
        input_ids = torch.cat([input_ids, torch.tensor([[best]])], dim=1)
    return "".join(TOKENS[i] for i in out)


def test_generates_valid_json_object() -> None:
    schema = {
        "type": "object",
        "properties": {"name": {"enum": ["a", "b"]}, "age": {"enum": [0, 1, 2]}},
        "required": ["name", "age"],
        "additionalProperties": False,
    }
    lp = GrammarLogitsProcessor.from_json_schema(schema, VOCAB)
    # hard-masking uses byte-level reachability, so greedy with a flat bias can
    # still trap itself in a partial-token dead end ("a" is a valid prefix of the
    # key "age" but no "ge" token exists). Bias toward the whole-word tokens.
    bias = torch.zeros(VOCAB.size)
    bias[TOKENS.index(" ")] = -5.0  # don't let flat-bias greedy loop on optional ws
    text = _greedy(lp, bias)
    obj = json.loads(text)  # must parse
    assert set(obj) == {"name", "age"}
    assert obj["name"] in ("a", "b")


def test_mask_matches_cfgconstraint_step_by_step() -> None:
    grammar = 'root ::= "[" ("a" | "b") ("," ("a" | "b"))* "]"'
    lp = GrammarLogitsProcessor(grammar, VOCAB)
    ref = CFGConstraint(grammar, VOCAB)

    input_ids = torch.zeros(1, 1, dtype=torch.long)
    for tok in [TOKENS.index(c) for c in ("[", "a", ",", "b", "]")]:
        scores = torch.zeros(1, VOCAB.size)
        lp(input_ids, scores)
        got = {i for i in range(VOCAB.size) if scores[0, i].item() == 0.0}
        want = {i for i in range(VOCAB.size) if ref.accepts(i)}
        assert got == want
        ref.advance(tok)
        input_ids = torch.cat([input_ids, torch.tensor([[tok]])], dim=1)
    lp(input_ids, torch.zeros(1, VOCAB.size))
    assert lp.is_complete()


def test_reset_and_reuse() -> None:
    lp = GrammarLogitsProcessor('root ::= "true" | "false"', VOCAB)
    a = _greedy(lp, torch.zeros(VOCAB.size))
    lp.reset()
    b = _greedy(lp, torch.zeros(VOCAB.size))
    assert a == b and a in ("true", "false")


def test_batch_rows_independent() -> None:
    lp = GrammarLogitsProcessor('root ::= "a"+ | "b"+', VOCAB)
    scores = torch.zeros(2, VOCAB.size)
    lp(torch.zeros(2, 1, dtype=torch.long), scores)
    # both rows start fresh: 'a' and 'b' allowed, digits not
    for r in (0, 1):
        assert scores[r, TOKENS.index("a")].item() == 0.0
        assert scores[r, TOKENS.index("0")].item() == float("-inf")
