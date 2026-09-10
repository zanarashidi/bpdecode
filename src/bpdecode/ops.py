"""``torch.ops.bpdecode.*`` -- the constrained-decoding kernels as torch ops.

Importing this module loads the compiled extension (``bpdecode._C``), which
registers the ops.  :class:`FsaTensors` is the tensor bundle the ops take,
built from a compiled DFA + :class:`~bpdecode.tokenizer.Vocabulary` (or the
host :class:`~bpdecode.fsa.FsaTable` / :class:`~bpdecode.fsa.TokenSymbols`).

CPU tensors run through the scalar core; CUDA tensors go to the device kernels
by raw pointer (move the bundle with :meth:`FsaTensors.to` first).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .fsa import FsaTable, TokenSymbols, fsa_from_dfa, token_symbols
from .regex import compile_regex
from .regex.compile import DFA
from .tokenizer import Vocabulary

try:
    from . import _C as _C  # noqa: F401  (dlopen side effect: registers torch.ops.bpdecode)
except ImportError as exc:  # pragma: no cover - build/packaging issue
    raise ImportError(
        "bpdecode._C is not built; reinstall with a C++ toolchain and torch "
        "available (`pip install -e .`)"
    ) from exc


@dataclass
class FsaTensors:
    """The FSA + token tables as tensors, the form the ops consume."""

    trans: torch.Tensor  # int32 [num_states * num_symbols]
    accept: torch.Tensor  # uint8 [num_states]
    live: torch.Tensor  # uint8 [num_states]
    num_symbols: int
    dead: int
    offsets: torch.Tensor  # int32 [vocab_size + 1]
    symbols: torch.Tensor  # int32 [offsets[-1]]
    eos_id: int
    start: int = 0
    accepting: frozenset[int] = frozenset()

    @property
    def device(self) -> torch.device:
        return self.trans.device

    def to(self, device: torch.device | str) -> FsaTensors:
        return FsaTensors(
            self.trans.to(device),
            self.accept.to(device),
            self.live.to(device),
            self.num_symbols,
            self.dead,
            self.offsets.to(device),
            self.symbols.to(device),
            self.eos_id,
            self.start,
            self.accepting,
        )

    @classmethod
    def from_tables(cls, fsa: FsaTable, toks: TokenSymbols) -> FsaTensors:
        return cls(
            trans=torch.tensor(fsa.trans, dtype=torch.int32),
            accept=torch.tensor(fsa.accept, dtype=torch.uint8),
            live=torch.tensor(fsa.live, dtype=torch.uint8),
            num_symbols=fsa.num_symbols,
            dead=fsa.dead,
            offsets=torch.tensor(toks.offsets, dtype=torch.int32),
            symbols=torch.tensor(toks.symbols or [0], dtype=torch.int32),
            eos_id=toks.eos_id,
            start=fsa.start,
            accepting=frozenset(i for i, a in enumerate(fsa.accept) if a),
        )

    @classmethod
    def build(cls, pattern: str | DFA, vocab: Vocabulary) -> FsaTensors:
        dfa = pattern if isinstance(pattern, DFA) else compile_regex(pattern)
        return cls.from_tables(fsa_from_dfa(dfa), token_symbols(dfa, vocab))


def apply_mask_(
    logits: torch.Tensor,
    fsa: FsaTensors,
    states: torch.Tensor,
    neg_inf: float = float("-inf"),
) -> torch.Tensor:
    """In place: set every disallowed logit in ``logits`` [batch, vocab] to
    ``neg_inf``, one grammar state per row.
    """
    return torch.ops.bpdecode.apply_mask_(
        logits, fsa.trans, fsa.accept, fsa.live, fsa.num_symbols, fsa.dead,
        fsa.offsets, fsa.symbols, fsa.eos_id, states, neg_inf,
    )


def compute_mask(fsa: FsaTensors, states: torch.Tensor) -> torch.Tensor:
    """Packed allow-mask, ``uint32`` bits in an int32 tensor [batch, ceil(vocab/32)]
    (LSB-first); one grammar state per row.
    """
    return torch.ops.bpdecode.compute_mask(
        fsa.trans, fsa.accept, fsa.live, fsa.num_symbols, fsa.dead,
        fsa.offsets, fsa.symbols, fsa.eos_id, states,
    )


def advance_state(
    fsa: FsaTensors, states: torch.Tensor, token_ids: torch.Tensor
) -> torch.Tensor:
    """Next grammar state per request after emitting ``token_ids``; ``-1`` if a
    sampled token was off a valid path.
    """
    return torch.ops.bpdecode.advance_state(
        fsa.trans, fsa.accept, fsa.live, fsa.num_symbols, fsa.dead,
        fsa.offsets, fsa.symbols, fsa.eos_id, states, token_ids,
    )
