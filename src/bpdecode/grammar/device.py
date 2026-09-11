"""Flatten a :class:`~bpdecode.grammar.pda.CompiledGrammar` into the tensor
bundle ``torch.ops.bpdecode.pda_*`` consumes -- a single global state space
(every rule's NFA states concatenated) with a flat CSR edge list, so a call
edge is just another state id to push.

This is the GPU path for the config-set PDA: :class:`~bpdecode.grammar.pda.PDA`
(host, unbounded, Python) is the reference; :class:`PdaTensors` bounds the
config-set to ``kPdaMaxConfigs`` alternative stacks of depth
``kPdaMaxDepth`` each (generous for JSON-Schema-scale grammars) so it fits
fixed-size device arrays. A grammar that needs more silently saturates --
`pda.hpp` documents this; it is not expected for realistic grammars.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .. import ops as _ops  # noqa: F401  (import side effect: registers torch.ops.bpdecode.*)
from ..tokenizer import Vocabulary
from .pda import CompiledGrammar

# Must match csrc/include/bpdecode/pda.hpp's kPdaMaxConfigs / kPdaMaxDepth.
PDA_MAX_CONFIGS = 8
PDA_MAX_DEPTH = 32
PDA_CONFIG_FLAT = 2 + PDA_MAX_CONFIGS + PDA_MAX_CONFIGS * PDA_MAX_DEPTH

_EPS, _BYTE, _CALL = 0, 1, 2


@dataclass
class PdaTensors:
    num_states: int
    root_start: int
    root_accept: int
    accept: torch.Tensor  # uint8 [num_states]
    live: torch.Tensor  # uint8 [num_states]
    edge_offsets: torch.Tensor  # int32 [num_states + 1]
    edge_kind: torch.Tensor  # int32 [nnz]
    edge_lo: torch.Tensor  # int32 [nnz]
    edge_hi: torch.Tensor  # int32 [nnz]
    edge_dst: torch.Tensor  # int32 [nnz]
    edge_callee: torch.Tensor  # int32 [nnz]
    tok_offsets: torch.Tensor  # int32 [vocab_size + 1]
    tok_bytes: torch.Tensor  # uint8 [tok_offsets[-1]]
    vocab_size: int
    eos_id: int

    @property
    def device(self) -> torch.device:
        return self.accept.device

    def to(self, device: torch.device | str) -> PdaTensors:
        if self.accept.device == torch.device(device):
            return self
        from dataclasses import replace

        fields = (
            "accept", "live", "edge_offsets", "edge_kind", "edge_lo",
            "edge_hi", "edge_dst", "edge_callee", "tok_offsets", "tok_bytes",
        )
        return replace(self, **{f: getattr(self, f).to(device) for f in fields})

    def init_config(self) -> torch.Tensor:
        """One freshly-initialised config-set, ``[PDA_CONFIG_FLAT]`` int32."""
        return torch.ops.bpdecode.pda_init(
            self.accept, self.live, self.edge_offsets, self.edge_kind,
            self.edge_lo, self.edge_hi, self.edge_dst, self.edge_callee,
            self.root_start, self.root_accept,
        )

    def init_batch(self, batch: int) -> torch.Tensor:
        """``batch`` copies of the initial config-set, ``[batch, PDA_CONFIG_FLAT]``."""
        return self.init_config().unsqueeze(0).expand(batch, -1).contiguous()


def build_pda_tensors(
    compiled: CompiledGrammar, vocab: Vocabulary, device: torch.device | str = "cpu"
) -> PdaTensors:
    order = list(compiled.rules)
    offset: dict[str, int] = {}
    total = 0
    for name in order:
        offset[name] = total
        total += compiled.rules[name].num_states

    accept = [0] * total
    live = [0] * total
    per_state: list[list[tuple[int, int, int, int, int]]] = [[] for _ in range(total)]

    for name in order:
        nfa = compiled.rules[name]
        base = offset[name]
        accept[base + nfa.accept] = 1
        for s in compiled.coreachable[name]:
            live[base + s] = 1
        for s in range(nfa.num_states):
            g = base + s
            for label, dst in nfa.out(s):
                if label is None:
                    per_state[g].append((_EPS, 0, 0, base + dst, -1))
                elif label[0] == "byte":
                    _, lo, hi = label
                    per_state[g].append((_BYTE, lo, hi, base + dst, -1))
                else:  # call
                    callee = label[1]
                    callee_start = offset[callee] + compiled.rules[callee].start
                    per_state[g].append((_CALL, 0, 0, base + dst, callee_start))

    edge_offsets = [0] * (total + 1)
    edge_kind: list[int] = []
    edge_lo: list[int] = []
    edge_hi: list[int] = []
    edge_dst: list[int] = []
    edge_callee: list[int] = []
    for s in range(total):
        edge_offsets[s] = len(edge_kind)
        for kind, lo, hi, dst, callee in per_state[s]:
            edge_kind.append(kind)
            edge_lo.append(lo)
            edge_hi.append(hi)
            edge_dst.append(dst)
            edge_callee.append(callee)
    edge_offsets[total] = len(edge_kind)

    root_start = offset[compiled.root] + compiled.rules[compiled.root].start
    root_accept = offset[compiled.root] + compiled.rules[compiled.root].accept

    tok_offsets = [0]
    tok_bytes = bytearray()
    for tb in vocab.token_bytes:
        tok_bytes.extend(tb)
        tok_offsets.append(len(tok_bytes))

    def t(data: list[int], dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(data, dtype=dtype, device=device)

    return PdaTensors(
        num_states=total,
        root_start=root_start,
        root_accept=root_accept,
        accept=t(accept, torch.uint8),
        live=t(live, torch.uint8),
        edge_offsets=t(edge_offsets, torch.int32),
        edge_kind=t(edge_kind or [0], torch.int32)[: len(edge_kind)],
        edge_lo=t(edge_lo or [0], torch.int32)[: len(edge_lo)],
        edge_hi=t(edge_hi or [0], torch.int32)[: len(edge_hi)],
        edge_dst=t(edge_dst or [0], torch.int32)[: len(edge_dst)],
        edge_callee=t(edge_callee or [0], torch.int32)[: len(edge_callee)],
        tok_offsets=t(tok_offsets, torch.int32),
        tok_bytes=torch.frombuffer(
            bytearray(tok_bytes) or bytearray(1), dtype=torch.uint8
        )[: len(tok_bytes)].clone().to(device),
        vocab_size=vocab.size,
        eos_id=vocab.eos_id if vocab.eos_id is not None else -1,
    )


def pda_apply_mask_(
    logits: torch.Tensor, g: PdaTensors, configs: torch.Tensor,
    neg_inf: float = float("-inf"),
) -> torch.Tensor:
    return torch.ops.bpdecode.pda_apply_mask_(
        logits, g.accept, g.live, g.edge_offsets, g.edge_kind, g.edge_lo,
        g.edge_hi, g.edge_dst, g.edge_callee, g.root_start, g.root_accept,
        g.tok_offsets, g.tok_bytes, g.eos_id, configs, neg_inf,
    )


def pda_advance_state(
    configs: torch.Tensor, g: PdaTensors, token_ids: torch.Tensor
) -> torch.Tensor:
    """Advance each config-set by its token in place; returns a uint8 ``ok``
    tensor (0 where the token was off a valid path -- that config-set is left
    unchanged, matching :class:`~bpdecode.grammar.pda.PDA`).
    """
    return torch.ops.bpdecode.pda_advance_state(
        configs, g.accept, g.live, g.edge_offsets, g.edge_kind, g.edge_lo,
        g.edge_hi, g.edge_dst, g.edge_callee, g.root_start, g.root_accept,
        g.tok_offsets, g.tok_bytes, g.eos_id, token_ids,
    )
