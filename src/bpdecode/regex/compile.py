"""Compile a regex AST into a deterministic finite automaton over UTF-8 bytes.

The pipeline is the textbook one: Thompson construction to an NFA with
epsilon moves, then subset construction to a DFA.  The alphabet is *bytes* --
``CharSet`` nodes carry Unicode code-point ranges, and :func:`_build_nfa`
lowers each one to its UTF-8 byte encoding (:mod:`bpdecode.regex.utf8`) so the
automaton consumes the same raw bytes the tokenizer emits.  Transitions are
keyed by *symbol classes* -- disjoint byte ranges that behave identically --
which keeps the table small; this is the CSR representation the CUDA kernels
consume, built and stored on the host for now.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .parser import (
    Alt,
    CharSet,
    Concat,
    Empty,
    Node,
    Opt,
    Plus,
    Star,
    parse,
)
from .utf8 import utf8_sequences

_BYTE_MAX = 0x100  # exclusive upper bound of the byte alphabet

Range = tuple[int, int]


@dataclass
class _NFA:
    start: int
    accept: int
    # state -> list of (ranges | None, target); None means an epsilon move
    edges: dict[int, list[tuple[tuple[Range, ...] | None, int]]] = field(default_factory=dict)
    _next: int = 0

    def new_state(self) -> int:
        s = self._next
        self._next += 1
        self.edges.setdefault(s, [])
        return s

    def add(self, src: int, ranges: tuple[Range, ...] | None, dst: int) -> None:
        self.edges.setdefault(src, []).append((ranges, dst))


def _build_nfa(node: Node) -> _NFA:
    nfa = _NFA(start=0, accept=0)

    def go(n: Node) -> tuple[int, int]:
        if isinstance(n, Empty):
            s = nfa.new_state()
            e = nfa.new_state()
            nfa.add(s, None, e)
            return s, e
        if isinstance(n, CharSet):
            s = nfa.new_state()
            e = nfa.new_state()
            # lower each code-point range to UTF-8: every byte sequence is a
            # fresh chain of single-byte-range edges from s to e.
            for lo, hi in n.ranges:
                for seq in utf8_sequences(lo, hi):
                    prev = s
                    for j, (blo, bhi) in enumerate(seq):
                        nxt = e if j == len(seq) - 1 else nfa.new_state()
                        nfa.add(prev, ((blo, bhi),), nxt)
                        prev = nxt
            return s, e
        if isinstance(n, Concat):
            s = e = None
            for part in n.parts:
                ps, pe = go(part)
                if s is None:
                    s, e = ps, pe
                else:
                    nfa.add(e, None, ps)
                    e = pe
            if s is None:  # empty Concat
                return go(Empty())
            return s, e
        if isinstance(n, Alt):
            s = nfa.new_state()
            e = nfa.new_state()
            for opt in n.options:
                os_, oe = go(opt)
                nfa.add(s, None, os_)
                nfa.add(oe, None, e)
            return s, e
        if isinstance(n, (Star, Plus, Opt)):
            inner_s, inner_e = go(n.node)
            s = nfa.new_state()
            e = nfa.new_state()
            nfa.add(s, None, inner_s)
            nfa.add(inner_e, None, e)
            if isinstance(n, (Star, Opt)):
                nfa.add(s, None, e)
            if isinstance(n, (Star, Plus)):
                nfa.add(inner_e, None, inner_s)
            return s, e
        raise TypeError(f"unhandled AST node: {type(n).__name__}")

    start, accept = go(node)
    nfa.start = start
    nfa.accept = accept
    return nfa


def _boundaries(nfa: _NFA) -> list[int]:
    """Return the sorted cut points that define the alphabet's symbol classes."""
    points = {0, _BYTE_MAX}
    for edge_list in nfa.edges.values():
        for ranges, _ in edge_list:
            if ranges is None:
                continue
            for lo, hi in ranges:
                points.add(lo)
                points.add(hi + 1)
    return sorted(points)


@dataclass(frozen=True)
class DFA:
    """A complete DFA over bytes.

    ``symbols`` are half-open ``[lo, hi)`` byte intervals; every byte in one
    interval drives the same transition.  ``trans[state][k]`` is the target of
    symbol class ``k`` (``-1`` = dead / no match).  ``start`` may be ``-1`` if
    the language is empty.
    """

    symbols: tuple[Range, ...]
    trans: tuple[tuple[int, ...], ...]
    accept: frozenset[int]
    start: int
    dead: int

    def symbol_of(self, byte: int) -> int:
        lo, hi = 0, len(self.symbols)
        while lo < hi:
            mid = (lo + hi) // 2
            s, e = self.symbols[mid]
            if byte < s:
                hi = mid
            elif byte >= e:
                lo = mid + 1
            else:
                return mid
        return -1

    def step(self, state: int, byte: int) -> int:
        if state < 0:
            return self.dead
        k = self.symbol_of(byte)
        if k < 0:
            return self.dead
        return self.trans[state][k]

    def live_states(self) -> frozenset[int]:
        """States from which some accepting state is reachable."""
        reverse: dict[int, set[int]] = {}
        for s, row in enumerate(self.trans):
            for t in row:
                if t >= 0:
                    reverse.setdefault(t, set()).add(s)
        seen = set(self.accept)
        stack = list(self.accept)
        while stack:
            cur = stack.pop()
            for pred in reverse.get(cur, ()):
                if pred not in seen:
                    seen.add(pred)
                    stack.append(pred)
        return frozenset(seen)


def _epsilon_closure(nfa: _NFA, states: frozenset[int]) -> frozenset[int]:
    stack = list(states)
    seen = set(states)
    while stack:
        s = stack.pop()
        for ranges, dst in nfa.edges.get(s, ()):
            if ranges is None and dst not in seen:
                seen.add(dst)
                stack.append(dst)
    return frozenset(seen)


def compile_regex(pattern: str) -> DFA:
    """Compile ``pattern`` to a :class:`DFA`. Convenience wrapper over :func:`compile_ast`."""
    return compile_ast(parse(pattern))


def compile_ast(ast: Node) -> DFA:
    nfa = _build_nfa(ast)
    cuts = _boundaries(nfa)
    symbols: tuple[Range, ...] = tuple(
        (cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)
    )
    sample = [lo for lo, _ in symbols]  # one representative code point per class

    start_set = _epsilon_closure(nfa, frozenset({nfa.start}))
    dfa_states: dict[frozenset[int], int] = {start_set: 0}
    order: list[frozenset[int]] = [start_set]
    rows: list[list[int]] = []

    i = 0
    while i < len(order):
        cur = order[i]
        row: list[int] = []
        for cp in sample:
            move: set[int] = set()
            for s in cur:
                for ranges, dst in nfa.edges.get(s, ()):
                    if ranges is None:
                        continue
                    if any(lo <= cp <= hi for lo, hi in ranges):
                        move.add(dst)
            if not move:
                row.append(-1)
                continue
            closure = _epsilon_closure(nfa, frozenset(move))
            idx = dfa_states.get(closure)
            if idx is None:
                idx = len(order)
                dfa_states[closure] = idx
                order.append(closure)
            row.append(idx)
        rows.append(row)
        i += 1

    # normalise: add an explicit dead state and point every -1 at it
    dead = len(order)
    for row in rows:
        for j, t in enumerate(row):
            if t < 0:
                row[j] = dead
    rows.append([dead] * len(symbols))  # dead state loops to itself
    full = rows

    accept = frozenset(
        idx for state_set, idx in dfa_states.items() if nfa.accept in state_set
    )
    return DFA(
        symbols=symbols,
        trans=tuple(tuple(r) for r in full),
        accept=accept,
        start=0,
        dead=dead,
    )
