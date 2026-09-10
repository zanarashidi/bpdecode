"""Hugging Face ``LogitsProcessor`` adapter.

``RegexLogitsProcessor`` drops into ``model.generate(..., logits_processor=[lp])``
and constrains the generated continuation to a regex.  It keeps a
:class:`~bpdecode.batch.ConstraintBatch` internally: one grammar state per logit
row, advanced by the token sampled at each step.

Greedy and sampling decoding are supported (row order is stable).  Beam search
reorders rows between steps and is not supported yet.
"""

from __future__ import annotations

import torch

from .batch import ConstraintBatch
from .ops import FsaTensors
from .regex.compile import DFA
from .tokenizer import Vocabulary


def _as_vocabulary(tokenizer_or_vocab: object) -> Vocabulary:
    if isinstance(tokenizer_or_vocab, Vocabulary):
        return tokenizer_or_vocab
    return Vocabulary.from_hf(tokenizer_or_vocab)


class RegexLogitsProcessor:
    """Constrain generation to ``pattern``. One instance per ``generate`` call."""

    def __init__(
        self,
        pattern: str | DFA,
        tokenizer: object,
        *,
        device: torch.device | str = "cpu",
        neg_inf: float = float("-inf"),
    ) -> None:
        vocab = _as_vocabulary(tokenizer)
        self._fsa: FsaTensors = FsaTensors.build(pattern, vocab).to(device)
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

        if scores.dtype == torch.float32:
            self._batch.apply_mask(self._rows, scores, self._neg_inf)
            return scores
        # some models hand the processor half-precision logits
        work = scores.float()
        self._batch.apply_mask(self._rows, work, self._neg_inf)
        scores.copy_(work)
        return scores

    def is_complete(self, row: int = 0) -> bool:
        """True when row ``row`` has reached an accepting state (EOS allowed)."""
        assert self._batch is not None, "call the processor at least once first"
        return self._batch.is_complete(self._rows[row])
