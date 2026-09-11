"""The constraint interface every backend implements.

A :class:`Constraint` tracks the grammar state for a *single* generation and
exposes three things the decoding loop needs:

* ``accepts(token_id)``   -- may this token be emitted next?
* ``advance(token_id)``   -- commit to a token, moving the state forward
* ``fill_mask(out)``      -- write the full allow-mask for the current state

The CPU reference (:mod:`bpdecode.reference`) implements this and is the
correctness oracle for every other backend.  ``__call__`` gives a
Hugging Face ``LogitsProcessor``-compatible shim so the same object can be
dropped into ``model.generate`` for end-to-end demos.

``as_logits_processor`` is a thin adapter that also handles batching (one
:class:`Constraint` per row).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class Constraint(Protocol):
    """Per-sequence grammar state."""

    @property
    def vocab_size(self) -> int: ...

    def reset(self) -> None: ...

    def accepts(self, token_id: int) -> bool: ...

    def advance(self, token_id: int) -> None: ...

    def is_complete(self) -> bool:
        """True when the string so far is a complete match (EOS is allowed)."""

    def fill_mask(self, out: MutableMaskLike) -> None:
        """Set ``out[i]`` truthy iff token ``i`` is currently allowed."""


class MutableMaskLike(Protocol):
    def __len__(self) -> int: ...
    def __setitem__(self, idx: int, value: object) -> None: ...


NEG_INF = -math.inf


class BaseConstraint(ABC):
    """Common machinery: mask -> logits bias, batching adapter, HF shim."""

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def accepts(self, token_id: int) -> bool: ...

    @abstractmethod
    def advance(self, token_id: int) -> None: ...

    @abstractmethod
    def is_complete(self) -> bool: ...

    @abstractmethod
    def allowed_ids(self) -> frozenset[int]: ...

    def fill_mask(self, out: MutableMaskLike) -> None:
        allowed = self.allowed_ids()
        for i in range(len(out)):
            out[i] = i in allowed

    def apply_(self, scores: list[float]) -> list[float]:
        """In-place: push disallowed logits to -inf. Works on any mutable sequence."""
        allowed = self.allowed_ids()
        for i in range(len(scores)):
            if i not in allowed:
                scores[i] = NEG_INF
        return scores

    # --- Hugging Face LogitsProcessor shim (single sequence) --------------
    def __call__(self, input_ids: Sequence[Sequence[int]], scores):  # noqa: ANN001
        try:
            import torch  # type: ignore

            if isinstance(scores, torch.Tensor):
                for row in range(scores.shape[0]):
                    allowed = self.allowed_ids()
                    mask = torch.ones(scores.shape[1], dtype=torch.bool)
                    if allowed:
                        mask[torch.tensor(sorted(allowed))] = False
                    scores[row][mask] = NEG_INF
                return scores
        except ImportError:
            pass
        for row in scores:
            self.apply_(row)
        return scores
