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
    # optional dense [num_states, vocab] int32: state after emitting token t from
    # state s, or -1 if t is rejected. When set, apply_mask_ / advance_state take
    # a gather fast path instead of re-walking token bytes.
    tok_next: torch.Tensor | None = None
    # optional soft-lookahead bias [num_states, vocab] float32 (see build_lookahead)
    lookahead: torch.Tensor | None = None

    @property
    def device(self) -> torch.device:
        return self.trans.device

    @property
    def num_states(self) -> int:
        return int(self.accept.numel())

    @property
    def vocab_size(self) -> int:
        return int(self.offsets.numel() - 1)

    def to(self, device: torch.device | str) -> FsaTensors:
        if self.trans.device == torch.device(device):
            return self
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
            None if self.tok_next is None else self.tok_next.to(device),
            None if self.lookahead is None else self.lookahead.to(device),
        )

    def densify(self) -> FsaTensors:
        """Return a copy carrying the precomputed ``tok_next`` table."""
        if self.tok_next is not None:
            return self
        from dataclasses import replace

        return replace(self, tok_next=build_token_transitions(self))

    def with_lookahead(self, k: int = 3) -> FsaTensors:
        """Return a copy carrying the ``k``-step soft-lookahead bias table."""
        from dataclasses import replace

        base = self.densify()
        return replace(base, lookahead=build_lookahead(base, k))

    @classmethod
    def from_tables(cls, fsa: FsaTable, toks: TokenSymbols) -> FsaTensors:
        return cls(
            trans=torch.tensor(fsa.trans, dtype=torch.int32),
            accept=torch.tensor(fsa.accept, dtype=torch.uint8),
            live=torch.tensor(fsa.live, dtype=torch.uint8),
            num_symbols=fsa.num_symbols,
            dead=fsa.dead,
            offsets=torch.tensor(toks.offsets, dtype=torch.int32),
            symbols=torch.frombuffer(
                bytearray(toks.symbols or b"\x00"), dtype=torch.uint8
            ).to(torch.int32),
            eos_id=toks.eos_id,
            start=fsa.start,
            accepting=frozenset(i for i, a in enumerate(fsa.accept) if a),
        )

    @classmethod
    def build(
        cls,
        pattern: str | DFA,
        vocab: Vocabulary,
        *,
        dense: bool | str = "auto",
    ) -> FsaTensors:
        dfa = pattern if isinstance(pattern, DFA) else compile_regex(pattern)
        fsa = cls.from_tables(fsa_from_dfa(dfa), token_symbols(dfa, vocab))
        want = dense is True or (
            dense == "auto" and fsa.num_states * fsa.vocab_size <= 32_000_000
        )
        return fsa.densify() if want else fsa


def build_token_transitions(
    fsa: FsaTensors, rows: torch.Tensor | list[int] | None = None
) -> torch.Tensor:
    """Dense ``[num_states, vocab]`` int32: state after emitting token ``t`` from
    state ``s`` (``-1`` if ``t`` is rejected -- undefined transition or a state
    no accepting state is reachable from). Matches ``step`` token for token.

    Built by running the whole vocab through the byte-DFA in parallel, one
    byte-position at a time. Pass ``rows`` to compute only those start states
    (the result is ``[len(rows), vocab]`` in that order).
    """
    dev = fsa.device
    S, M, V = fsa.num_states, fsa.num_symbols, fsa.vocab_size
    trans = fsa.trans.to(torch.long)
    offsets = fsa.offsets.to(torch.long)
    lengths = offsets[1:] - offsets[:-1]  # [V]

    row_states = (
        torch.arange(S, device=dev)
        if rows is None
        else torch.as_tensor(rows, device=dev, dtype=torch.long)
    )
    R = row_states.numel()
    cur = row_states.view(R, 1).expand(R, V).contiguous().to(torch.long)
    if V and int(lengths.max()):
        sym = fsa.symbols.to(torch.long)  # [nnz]
        # walk tokens shortest-first so iteration k only touches the tokens
        # still mid-walk (a contiguous tail) -- total work is sum(lengths), not V*maxlen
        order = torch.argsort(lengths)
        slen = lengths[order]
        soff = offsets[:-1][order]
        cur = cur[:, order]
        for k in range(int(slen[-1])):
            first = int(torch.searchsorted(slen, k + 1))
            if first >= V:
                break
            seg = slice(first, V)
            sym_k = sym[soff[seg] + k].view(1, -1)
            cur[:, seg] = trans[cur[:, seg] * M + sym_k]
        inv = torch.empty_like(order)
        inv[order] = torch.arange(V, device=dev)
        cur = cur[:, inv].contiguous()

    live = fsa.live.to(torch.bool)
    out = torch.where(
        live[cur], cur.to(torch.int32), torch.full_like(cur, -1, dtype=torch.int32)
    )
    if fsa.eos_id is not None and 0 <= fsa.eos_id < V:
        acc = fsa.accept.to(torch.bool)[row_states]  # [R]
        out[:, fsa.eos_id] = torch.where(
            acc,
            row_states.to(torch.int32),
            torch.full((R,), -1, dtype=torch.int32, device=dev),
        )
    return out


def build_lookahead(fsa: FsaTensors, k: int) -> torch.Tensor:
    """Soft-lookahead bias table ``[num_states, vocab]`` (float32).

    ``out[s, t]`` is ``log Z_{k-1}(delta(s, t))`` -- the log of how many
    grammar-valid token strings of length ``k - 1`` can follow token ``t``
    emitted from state ``s`` (``-inf`` if ``t`` is not allowed from ``s``).
    Larger = the token keeps more of the language open; add ``alpha * out`` to
    the logits to steer away from valid-but-dead-end tokens (and it masks for
    free, since disallowed entries are ``-inf``).

    ``Z_j`` is the backward sum-product recurrence -- the weighted-count analogue
    of ``build_reachability``'s boolean fixpoint -- run in the log domain:
        logZ_0(s)   = 0 if s is live else -inf
        logZ_{j+1}(s) = logsumexp_t logZ_j(next(s, t))
    where ``next(s, EOS) = DONE`` with ``logZ_j(DONE) = 0`` when s is accepting.
    """
    if fsa.tok_next is None:
        fsa = fsa.densify()
    tn = fsa.tok_next.to(torch.long)  # [S, V]
    S, V = tn.shape
    dev = tn.device
    live = fsa.live.to(torch.bool)
    accept = fsa.accept.to(torch.bool)
    neg_inf = torch.tensor(float("-inf"), device=dev)

    logz = torch.where(live, torch.zeros(S, device=dev), neg_inf.expand(S))  # [S]
    for _ in range(max(k - 1, 0)):
        cand = torch.where(tn >= 0, logz[tn.clamp(min=0)], neg_inf)  # [S, V]
        nxt = torch.logsumexp(cand, dim=1)  # [S]
        # EOS from an accepting state reaches DONE (logZ = 0)
        nxt = torch.where(accept, torch.logaddexp(nxt, torch.zeros(S, device=dev)), nxt)
        logz = torch.where(live, nxt, neg_inf.expand(S))

    out = torch.where(tn >= 0, logz[tn.clamp(min=0)], neg_inf)  # [S, V]
    if fsa.eos_id is not None and 0 <= fsa.eos_id < V:
        out[:, fsa.eos_id] = torch.where(accept, torch.zeros(S, device=dev), neg_inf)
    return out


def apply_soft_(
    logits: torch.Tensor,
    fsa: FsaTensors,
    states: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """In place: ``logits += alpha * lookahead[states]``.

    Disallowed tokens get ``-inf`` regardless of ``alpha`` (so this also masks);
    ``alpha = 0`` is exactly hard masking.
    """
    if fsa.lookahead is None:
        raise ValueError("fsa has no lookahead table; build with dense + with_lookahead")
    s = states.long()
    bias = fsa.lookahead.index_select(0, s.clamp(min=0)).clone()
    bias[s < 0] = float("-inf")  # BROKEN / FREE
    if alpha == 0.0:
        return logits.masked_fill_(torch.isneginf(bias), float("-inf"))
    return logits.add_(bias, alpha=alpha)


def apply_mask_(
    logits: torch.Tensor,
    fsa: FsaTensors,
    states: torch.Tensor,
    neg_inf: float = float("-inf"),
) -> torch.Tensor:
    """In place: set every disallowed logit in ``logits`` [batch, vocab] to
    ``neg_inf``, one grammar state per row.
    """
    if fsa.tok_next is not None:
        s = states.long()
        broken = s < 0  # BROKEN / FREE -> nothing is allowed
        rejected = fsa.tok_next.index_select(0, s.clamp_(min=0)) == -1  # [B, V]
        rejected |= broken.unsqueeze(1)
        return logits.masked_fill_(rejected, neg_inf)
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
    if fsa.tok_next is not None:
        s = states.long()
        t = token_ids.long()
        broken = s < 0
        nxt = fsa.tok_next[s.clamp_(min=0), t]
        return torch.where(broken, torch.full_like(nxt, -1), nxt).to(torch.int32)
    return torch.ops.bpdecode.advance_state(
        fsa.trans, fsa.accept, fsa.live, fsa.num_symbols, fsa.dead,
        fsa.offsets, fsa.symbols, fsa.eos_id, states, token_ids,
    )
