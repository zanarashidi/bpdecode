"""Host-side FSA tables in the exact shape the C++/CUDA core consumes.

:class:`FsaTable` and :class:`TokenSymbols` mirror the structs in
``csrc/include/bpdecode/mask.hpp`` field for field.  The host front-end builds
them from a compiled :class:`~bpdecode.regex.compile.DFA` and a
:class:`~bpdecode.tokenizer.Vocabulary`; Phase 1 uploads them to the device as
CSR / SoA arrays.

Two builders:

* :func:`fsa_from_dfa` -- dense transition table + backward-reachability
  (``live``) solve.  The reachability pass is the boolean message-passing
  fixpoint carried over from the BP kernels; :func:`build_reachability` exposes
  it standalone so the CUDA port has a scalar oracle.
* :func:`token_symbols` -- ragged token id -> symbol-class-id sequence.

:func:`compute_mask` / :func:`step` re-implement the C++ scalar semantics in
Python so the export can be differential-tested against
:class:`~bpdecode.automaton.TokenDFA` without a compiler in the loop.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .regex.compile import DFA
from .tokenizer import Vocabulary


@dataclass(frozen=True)
class FsaTable:
    """Dense DFA over symbol classes, row-major ``trans[state * num_symbols + sym]``.

    ``dead`` is absorbing and non-accepting; ``live[s]`` is true iff an accepting
    state is reachable from ``s``.  ``symbols`` keeps the half-open code-point
    ranges each class stands for -- needed to map token bytes to class ids, not
    uploaded to the device.
    """

    num_states: int
    num_symbols: int
    start: int
    dead: int
    trans: tuple[int, ...]  # num_states * num_symbols
    accept: tuple[int, ...]  # num_states, 0/1
    live: tuple[int, ...]  # num_states, 0/1
    symbols: tuple[tuple[int, int], ...] = ()  # num_symbols, [lo, hi)

    def target(self, state: int, sym: int) -> int:
        return self.trans[state * self.num_symbols + sym]


@dataclass
class TokenSymbols:
    """Token id -> the symbol-class ids its code points drive through the FSA.

    Ragged: token ``t`` occupies ``symbols[offsets[t]:offsets[t + 1]]``.  A code
    point outside every class is stored as ``-1``; the scalar ``step`` treats
    such a token as never allowed.
    """

    vocab_size: int
    eos_id: int
    offsets: list[int] = field(default_factory=lambda: [0])
    symbols: list[int] = field(default_factory=list)


def build_reachability(
    num_states: int,
    num_symbols: int,
    trans: list[int] | tuple[int, ...],
    accept: list[int] | tuple[int, ...],
) -> list[int]:
    """Backward-reachability solve: ``live[s] == 1`` iff some accepting state is
    reachable from ``s``.

    Boolean message passing to a fixpoint -- here as a reverse BFS from the
    accepting set, which is the same fixpoint reached in one linear sweep.  The
    CUDA port iterates ``live |= OR(live[succ])`` until stable; this is its
    reference.
    """
    preds: list[list[int]] = [[] for _ in range(num_states)]
    for s in range(num_states):
        base = s * num_symbols
        for k in range(num_symbols):
            t = trans[base + k]
            if 0 <= t < num_states:
                preds[t].append(s)

    live = [0] * num_states
    queue: deque[int] = deque()
    for s in range(num_states):
        if accept[s]:
            live[s] = 1
            queue.append(s)
    while queue:
        cur = queue.popleft()
        for p in preds[cur]:
            if not live[p]:
                live[p] = 1
                queue.append(p)
    return live


def fsa_from_dfa(dfa: DFA) -> FsaTable:
    """Flatten a :class:`DFA` into a dense :class:`FsaTable` and solve ``live``.

    The :class:`DFA` already carries an explicit absorbing ``dead`` state with
    every missing transition pointed at it, so this is a straight flatten plus
    the reachability pass.
    """
    num_states = len(dfa.trans)
    num_symbols = len(dfa.symbols)
    flat: list[int] = []
    for row in dfa.trans:
        flat.extend(row)
    accept = [1 if s in dfa.accept else 0 for s in range(num_states)]
    live = build_reachability(num_states, num_symbols, flat, accept)
    return FsaTable(
        num_states=num_states,
        num_symbols=num_symbols,
        start=dfa.start,
        dead=dfa.dead,
        trans=tuple(flat),
        accept=tuple(accept),
        live=tuple(live),
        symbols=tuple(dfa.symbols),
    )


def token_symbols(dfa: DFA, vocab: Vocabulary) -> TokenSymbols:
    """Map every token id to the sequence of :class:`DFA` symbol classes its
    code points select.  Code points outside every class become ``-1``.
    """
    offsets = [0]
    syms: list[int] = []
    for tb in vocab.token_bytes:
        for ch in tb.decode("utf-8", "surrogateescape"):
            syms.append(dfa.symbol_of(ord(ch)))
        offsets.append(len(syms))
    eos = vocab.eos_id if vocab.eos_id is not None else -1
    return TokenSymbols(
        vocab_size=vocab.size, eos_id=eos, offsets=offsets, symbols=syms
    )


def step(fsa: FsaTable, toks: TokenSymbols, state: int, token_id: int) -> int:
    """Scalar next-state, mirroring ``bpdecode::step`` in ``mask_cpu.cpp``.

    Returns the state after emitting ``token_id`` from ``state``, or ``-1`` if
    the token is not accepted (undefined transition, or lands somewhere no
    accepting state is reachable from).
    """
    if token_id == toks.eos_id:
        return state if (state >= 0 and fsa.accept[state]) else -1
    if state < 0:
        return -1
    cur = state
    for i in range(toks.offsets[token_id], toks.offsets[token_id + 1]):
        sym = toks.symbols[i]
        if sym < 0 or sym >= fsa.num_symbols:
            return -1
        cur = fsa.trans[cur * fsa.num_symbols + sym]
        if cur == fsa.dead:
            return -1
    return cur if fsa.live[cur] else -1


def compute_mask(fsa: FsaTable, toks: TokenSymbols, state: int) -> list[bool]:
    """Boolean allow-mask of length ``vocab_size`` for ``state`` -- mirrors
    ``bpdecode::compute_mask``.
    """
    out = [False] * toks.vocab_size
    for t in range(toks.vocab_size):
        if step(fsa, toks, state, t) != -1:
            out[t] = True
    if toks.eos_id >= 0 and state >= 0 and fsa.accept[state]:
        out[toks.eos_id] = True
    return out
