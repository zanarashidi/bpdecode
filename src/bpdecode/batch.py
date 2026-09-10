"""Batched constraint state for continuous-batching serving.

A server runs many sequences at once, joining and leaving the batch every step.
:class:`ConstraintBatch` keeps one grammar state per active sequence in a single
device tensor, so advancing the whole batch and masking the whole batch are each
one kernel launch (via :mod:`bpdecode.ops`).

:class:`GrammarCache` compiles each distinct pattern once and hands the shared
:class:`~bpdecode.ops.FsaTensors` to every sequence that uses it -- "shared
compiled DFA across identical grammars".

Phase 2 scope: a single grammar per batch (the common case -- every request
against the same JSON schema). Mixed-grammar batches are grouped by the caller
for now; native grouping is a follow-up.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable, Sequence

import torch

from .ops import FsaTensors, advance_state, apply_mask_
from .tokenizer import Vocabulary

FREE = -2  # slot holds no sequence
BROKEN = -1  # a committed token left the grammar-valid path (matches step() == -1)


class GrammarCache:
    """LRU cache of compiled grammars for one tokenizer / device."""

    def __init__(
        self,
        vocab: Vocabulary,
        device: torch.device | str = "cpu",
        max_size: int = 128,
    ) -> None:
        self._vocab = vocab
        self._device = torch.device(device)
        self._max = max_size
        self._cache: OrderedDict[str, FsaTensors] = OrderedDict()

    def get(self, pattern: str) -> FsaTensors:
        hit = self._cache.get(pattern)
        if hit is not None:
            self._cache.move_to_end(pattern)
            return hit
        fsa = FsaTensors.build(pattern, self._vocab).to(self._device)
        self._cache[pattern] = fsa
        if len(self._cache) > self._max:
            self._cache.popitem(last=False)
        return fsa

    def __len__(self) -> int:
        return len(self._cache)


class ConstraintBatch:
    """A fixed-capacity pool of grammar states, all sharing one ``FsaTensors``.

    Slots are stable for a sequence's lifetime (``add`` -> ``evict``); the caller
    passes an ordered list of sequence ids each step that lines up with its logit
    rows.
    """

    def __init__(
        self,
        fsa: FsaTensors,
        capacity: int,
        device: torch.device | str | None = None,
    ) -> None:
        self.fsa = fsa if device is None else fsa.to(device)
        self.device = self.fsa.device
        self.capacity = capacity
        self._state = torch.full(
            (capacity,), FREE, dtype=torch.int32, device=self.device
        )
        self._slot_of: dict[Hashable, int] = {}
        self._free: list[int] = list(reversed(range(capacity)))

    # --- lifecycle -------------------------------------------------------
    def add(self, seq_id: Hashable) -> int:
        if seq_id in self._slot_of:
            raise KeyError(f"sequence {seq_id!r} already in the batch")
        if not self._free:
            raise RuntimeError("ConstraintBatch is at capacity")
        slot = self._free.pop()
        self._slot_of[seq_id] = slot
        self._state[slot] = self.fsa.start
        return slot

    def evict(self, seq_id: Hashable) -> None:
        slot = self._slot_of.pop(seq_id)
        self._state[slot] = FREE
        self._free.append(slot)

    def reset(self, seq_id: Hashable) -> None:
        self._state[self._slot_of[seq_id]] = self.fsa.start

    def __len__(self) -> int:
        return len(self._slot_of)

    def __contains__(self, seq_id: Hashable) -> bool:
        return seq_id in self._slot_of

    # --- per-step ------------------------------------------------------
    def _slots(self, seq_ids: Sequence[Hashable]) -> torch.Tensor:
        return torch.tensor(
            [self._slot_of[s] for s in seq_ids], dtype=torch.long, device=self.device
        )

    def apply_mask(
        self,
        seq_ids: Sequence[Hashable],
        logits: torch.Tensor,
        neg_inf: float = float("-inf"),
    ) -> torch.Tensor:
        """In place: push every grammar-disallowed logit in ``logits``
        ``[len(seq_ids), vocab]`` to ``neg_inf``, one row per ``seq_ids`` entry.
        """
        states = self._state[self._slots(seq_ids)]
        apply_mask_(logits, self.fsa, states, neg_inf)
        return logits

    def commit(
        self, seq_ids: Sequence[Hashable], tokens: torch.Tensor
    ) -> torch.Tensor:
        """Advance each named sequence by the token it sampled. Returns the new
        state per sequence (``BROKEN`` if the token was off a valid path).
        """
        slots = self._slots(seq_ids)
        nxt = advance_state(self.fsa, self._state[slots], tokens.to(torch.int32))
        self._state[slots] = nxt
        return nxt

    # --- queries ------------------------------------------------------
    def state_of(self, seq_id: Hashable) -> int:
        return int(self._state[self._slot_of[seq_id]])

    def is_complete(self, seq_id: Hashable) -> bool:
        """True when the string so far is a full match (EOS is allowed)."""
        return self.state_of(seq_id) in self.fsa.accepting

    def is_broken(self, seq_id: Hashable) -> bool:
        return self.state_of(seq_id) == BROKEN
