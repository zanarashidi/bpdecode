"""Splice a regular sub-grammar out of a PDA frame and compile it to a byte DFA.

The slow config-sets in :class:`~bpdecode.grammar.pda.PDA` are the ones sitting
in a regular loop -- ``json-char*``, ``[0-9]+``, whitespace.  For such a top
frame the residual language *until the frame pops* is regular (once regular
callees are inlined), so it compiles to a :class:`~bpdecode.regex.compile.DFA`
and the mask comes from the dense ``tok_next`` tensor path instead of a
per-token walk.

``residual_dfa`` returns that DFA with the rule's accept states preserved as
the DFA accept set -- those mark where the frame could pop (a *boundary*, which
still needs the real stack), everything else is context-independent.
"""

from __future__ import annotations

from ..regex.compile import DFA
from .nfa import RuleNFA
from .pda import CompiledGrammar


def regular_rules(compiled: CompiledGrammar) -> frozenset[str]:
    """Rules that (transitively) call only regular rules."""
    calls = {
        name: {
            lbl[1]
            for s in range(nfa.num_states)
            for lbl, _ in nfa.out(s)
            if lbl is not None and lbl[0] == "call"
        }
        for name, nfa in compiled.rules.items()
    }
    regular = {name for name, c in calls.items() if not c}
    changed = True
    while changed:
        changed = False
        for name, c in calls.items():
            if name not in regular and c <= regular:
                regular.add(name)
                changed = True
    return frozenset(regular)


def _reachable_no_pop(nfa: RuleNFA, start: int) -> set[int]:
    seen = {start}
    stack = [start]
    while stack:
        for _, dst in nfa.out(stack.pop()):
            if dst not in seen:
                seen.add(dst)
                stack.append(dst)
    return seen


def frame_is_regular(
    compiled: CompiledGrammar, regular: frozenset[str], rule: str, state: int
) -> bool:
    """Can the residual of frame ``(rule, state)`` be compiled to a DFA?"""
    nfa = compiled.rules[rule]
    for s in _reachable_no_pop(nfa, state):
        for lbl, _ in nfa.out(s):
            if lbl is not None and lbl[0] == "call" and lbl[1] not in regular:
                return False
    return True


class _Splicer:
    """Flatten frame (rule, state) + inlined regular callees into one byte NFA."""

    def __init__(self, compiled: CompiledGrammar) -> None:
        self.c = compiled
        self.edges: dict[int, list[tuple[object, int]]] = {}
        self._n = 0

    def new(self) -> int:
        s = self._n
        self._n += 1
        self.edges.setdefault(s, [])
        return s

    def add(self, src: int, label: object, dst: int) -> None:
        self.edges.setdefault(src, []).append((label, dst))

    def inline(self, rule: str, start_nfa: int) -> tuple[int, int]:
        """Return (start, accept) in the spliced NFA for `rule` entered at
        `start_nfa`."""
        nfa = self.c.rules[rule]
        local: dict[int, int] = {}

        def get(orig: int) -> int:
            if orig not in local:
                local[orig] = self.new()
            return local[orig]

        for s in range(nfa.num_states):
            src = get(s)
            for lbl, dst in nfa.out(s):
                if lbl is None:
                    self.add(src, None, get(dst))
                elif lbl[0] == "byte":
                    self.add(src, lbl, get(dst))
                else:  # call -> inline (callee is regular by precondition)
                    cs, ce = self.inline(lbl[1], self.c.rules[lbl[1]].start)
                    self.add(src, None, cs)
                    self.add(ce, None, get(dst))
        return get(start_nfa), get(nfa.accept)


_BYTE_SYMBOLS = tuple((b, b + 1) for b in range(256))


def _subset_construct(
    edges: dict[int, list[tuple[object, int]]], start: int, accept: int
) -> DFA:
    # 256 byte-indexed classes so every residual DFA shares one token->symbol
    # mapping (see grammar.constraint._masks_for_dfa) -- residual DFAs are tiny.
    symbols = _BYTE_SYMBOLS
    reps = list(range(256))

    def eclose(states: frozenset[int]) -> frozenset[int]:
        seen = set(states)
        stack = list(states)
        while stack:
            for lbl, dst in edges.get(stack.pop(), ()):
                if lbl is None and dst not in seen:
                    seen.add(dst)
                    stack.append(dst)
        return frozenset(seen)

    start_set = eclose(frozenset({start}))
    order = [start_set]
    index = {start_set: 0}
    rows: list[list[int]] = []
    i = 0
    while i < len(order):
        cur = order[i]
        row: list[int] = []
        for b in reps:
            move: set[int] = set()
            for s in cur:
                for lbl, dst in edges.get(s, ()):
                    if lbl is not None and lbl[1] <= b <= lbl[2]:
                        move.add(dst)
            if not move:
                row.append(-1)
                continue
            closure = eclose(frozenset(move))
            j = index.get(closure)
            if j is None:
                j = len(order)
                index[closure] = j
                order.append(closure)
            row.append(j)
        rows.append(row)
        i += 1

    dead = len(order)
    for row in rows:
        for k, t in enumerate(row):
            if t < 0:
                row[k] = dead
    rows.append([dead] * len(symbols))
    accepts = frozenset(idx for st, idx in index.items() if accept in st)
    return DFA(
        symbols=symbols,
        trans=tuple(tuple(r) for r in rows),
        accept=accepts,
        start=0,
        dead=dead,
    )


def _eclose_in_rule(nfa: RuleNFA, state: int) -> frozenset[int]:
    seen = {state}
    stack = [state]
    while stack:
        for lbl, dst in nfa.out(stack.pop()):
            if lbl is None and dst not in seen:
                seen.add(dst)
                stack.append(dst)
    return frozenset(seen)


def residual_dfa(compiled: CompiledGrammar, rule: str, state: int) -> DFA:
    """Byte DFA for frame ``(rule, state)``'s residual language; DFA accept
    states are where the frame could pop (a boundary).

    States with the same epsilon-closure share a residual, so the DFA is cached
    by ``(rule, closure)`` -- collapses the many NFA states of one loop.
    """
    key = (rule, _eclose_in_rule(compiled.rules[rule], state))
    cache = compiled.residual_memo
    hit = cache.get(("dfa", key))
    if hit is None:
        sp = _Splicer(compiled)
        s, a = sp.inline(rule, state)
        hit = _subset_construct(sp.edges, s, a)
        cache[("dfa", key)] = hit
    return hit
