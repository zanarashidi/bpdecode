"""Hugging Face ``LogitsProcessor`` adapters.

``RegexLogitsProcessor`` constrains generation to a regex via the batched
tensor path (:class:`~bpdecode.batch.ConstraintBatch`).  ``GrammarLogitsProcessor``
constrains it to a context-free grammar or JSON Schema via the per-row CPU
:class:`~bpdecode.grammar.constraint.CFGConstraint` (masks are memoised on the
compiled grammar, so the first constrained generation warms the cache and the
rest are cheap).

Both drop into ``model.generate(..., logits_processor=[lp])``.  Greedy and
sampling decoding are supported; beam search reorders rows and is not.
"""

from __future__ import annotations

import torch

from .batch import ConstraintBatch
from .grammar.constraint import CFGConstraint
from .grammar.gbnf import parse_gbnf
from .grammar.ir import Grammar
from .grammar.json_schema import json_schema_to_grammar
from .grammar.pda import CompiledGrammar
from .ops import FsaTensors
from .regex.compile import DFA
from .tokenizer import Vocabulary


def _as_vocabulary(tokenizer_or_vocab: object) -> Vocabulary:
    if isinstance(tokenizer_or_vocab, Vocabulary):
        return tokenizer_or_vocab
    return Vocabulary.from_hf(tokenizer_or_vocab)


class RegexLogitsProcessor:
    """Constrain generation to ``pattern``. One instance per ``generate`` call.

    With ``soft_k`` set, hard masking is replaced by a k-step soft-lookahead
    bias (``alpha`` scales it; ``alpha=0`` is plain masking) -- steers away from
    valid-but-dead-end tokens and prunes tokens with no valid token continuation.
    """

    def __init__(
        self,
        pattern: str | DFA,
        tokenizer: object,
        *,
        device: torch.device | str = "cpu",
        neg_inf: float = float("-inf"),
        soft_k: int | None = None,
        alpha: float = 1.0,
    ) -> None:
        vocab = _as_vocabulary(tokenizer)
        fsa = FsaTensors.build(pattern, vocab)
        self._soft = soft_k is not None
        self._alpha = alpha
        if self._soft:
            fsa = fsa.with_lookahead(soft_k)
        self._fsa: FsaTensors = fsa.to(device)
        self._neg_inf = neg_inf
        self._batch: ConstraintBatch | None = None
        self._rows: list[int] = []

    def reset(self) -> None:
        """Forget accumulated state so the processor can be reused."""
        self._batch = None
        self._rows = []

    def __call__(
        self, input_ids: torch.Tensor, scores: torch.Tensor
    ) -> torch.Tensor:
        n_rows = scores.shape[0]

        if self._batch is None:
            self._batch = ConstraintBatch(
                self._fsa, capacity=n_rows, device=scores.device
            )
            self._rows = list(range(n_rows))
            for r in self._rows:
                self._batch.add(r)
        else:
            if n_rows != len(self._rows):
                raise RuntimeError(
                    "RegexLogitsProcessor: row count changed between steps "
                    "(beam search is not supported); call reset() per generation"
                )
            last = input_ids[:, -1]
            self._batch.commit(self._rows, last)

        v = self._fsa.vocab_size

        def _mask(x: torch.Tensor) -> None:
            # a model may have more logit columns than real tokens (padded
            # lm_head); those extras can never be valid.
            if x.shape[1] > v:
                x[:, v:] = self._neg_inf
            view = x[:, :v]
            if self._soft:
                self._batch.apply_soft(self._rows, view, self._alpha)
            else:
                self._batch.apply_mask(self._rows, view, self._neg_inf)

        if scores.dtype == torch.float32:
            _mask(scores)
            return scores
        # some models hand the processor half-precision logits
        work = scores.float()
        _mask(work)
        scores.copy_(work)
        return scores

    def is_complete(self, row: int = 0) -> bool:
        """True when row ``row`` has reached an accepting state (EOS allowed)."""
        assert self._batch is not None, "call the processor at least once first"
        return self._batch.is_complete(self._rows[row])


class GrammarLogitsProcessor:
    """Constrain generation to a CFG. One instance per ``generate`` call.

    ``grammar`` is GBNF source, a :class:`Grammar`, or -- via
    :meth:`from_json_schema` -- a JSON Schema.  A regular grammar still works
    here but :class:`RegexLogitsProcessor` is far faster for those.
    """

    def __init__(
        self,
        grammar: str | Grammar,
        tokenizer: object,
        *,
        root: str = "root",
        neg_inf: float = float("-inf"),
    ) -> None:
        g = grammar if isinstance(grammar, Grammar) else parse_gbnf(grammar, root)
        self._vocab = _as_vocabulary(tokenizer)
        self._compiled = CompiledGrammar.build(g)
        self._neg_inf = neg_inf
        self._cons: list[CFGConstraint] = []

    @classmethod
    def from_json_schema(
        cls, schema: object, tokenizer: object, **kw: object
    ) -> GrammarLogitsProcessor:
        return cls(json_schema_to_grammar(schema), tokenizer, **kw)  # type: ignore[arg-type]

    def reset(self) -> None:
        self._cons = []

    def __call__(
        self, input_ids: torch.Tensor, scores: torch.Tensor
    ) -> torch.Tensor:
        n_rows = scores.shape[0]
        if not self._cons:
            self._cons = [
                CFGConstraint.from_compiled(self._compiled, self._vocab)
                for _ in range(n_rows)
            ]
        else:
            if n_rows != len(self._cons):
                raise RuntimeError(
                    "GrammarLogitsProcessor: row count changed (beam search "
                    "unsupported); call reset() per generation"
                )
            last = input_ids[:, -1].tolist()
            for con, tok in zip(self._cons, last, strict=True):
                con.advance(int(tok))

        vocab_size = scores.shape[1]
        for row, con in enumerate(self._cons):
            allowed = con.allowed_ids()
            block = torch.ones(vocab_size, dtype=torch.bool, device=scores.device)
            if allowed:
                idx = torch.tensor(sorted(allowed), device=scores.device)
                block[idx] = False
            scores[row][block] = self._neg_inf
        return scores

    def is_complete(self, row: int = 0) -> bool:
        return bool(self._cons) and self._cons[row].is_complete()
