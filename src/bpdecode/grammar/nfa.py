"""Compile each grammar rule to a byte-level NFA whose edges are either a byte
range (a terminal) or a call to another rule (a nonterminal).

Thompson construction, same as the regex compiler, plus a ``Call`` edge.  Code
points in ``CharSet`` nodes are lowered to their UTF-8 byte sequences via
:func:`bpdecode.regex.utf8.utf8_sequences`, so the whole machine runs on bytes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..regex.utf8 import utf8_sequences
from .ir import Alt, CharSet, Concat, Empty, Expr, Grammar, Opt, Plus, Ref, Star

# edge label: None = epsilon, ("byte", lo, hi), ("call", rule_name)
Label = None | tuple


@dataclass
class RuleNFA:
    start: int
    accept: int
    num_states: int
    edges: dict[int, list[tuple[Label, int]]] = field(default_factory=dict)

    def out(self, s: int) -> list[tuple[Label, int]]:
        return self.edges.get(s, [])


class _Builder:
    def __init__(self) -> None:
        self.edges: dict[int, list[tuple[Label, int]]] = {}
        self._n = 0

    def state(self) -> int:
        s = self._n
        self._n += 1
        self.edges.setdefault(s, [])
        return s

    def add(self, src: int, label: Label, dst: int) -> None:
        self.edges.setdefault(src, []).append((label, dst))

    def build(self, expr: Expr) -> tuple[int, int]:
        if isinstance(expr, Empty):
            s = self.state()
            e = self.state()
            self.add(s, None, e)
            return s, e
        if isinstance(expr, Ref):
            s = self.state()
            e = self.state()
            self.add(s, ("call", expr.name), e)
            return s, e
        if isinstance(expr, CharSet):
            s = self.state()
            e = self.state()
            for lo, hi in expr.ranges:
                for seq in utf8_sequences(lo, hi):
                    prev = s
                    for j, (blo, bhi) in enumerate(seq):
                        nxt = e if j == len(seq) - 1 else self.state()
                        self.add(prev, ("byte", blo, bhi), nxt)
                        prev = nxt
            return s, e
        if isinstance(expr, Concat):
            s = e = None
            for part in expr.parts:
                ps, pe = self.build(part)
                if s is None:
                    s, e = ps, pe
                else:
                    self.add(e, None, ps)
                    e = pe
            if s is None:
                return self.build(Empty())
            return s, e
        if isinstance(expr, Alt):
            s = self.state()
            e = self.state()
            for opt in expr.options:
                os_, oe = self.build(opt)
                self.add(s, None, os_)
                self.add(oe, None, e)
            return s, e
        if isinstance(expr, (Star, Plus, Opt)):
            inner_s, inner_e = self.build(expr.node)
            s = self.state()
            e = self.state()
            self.add(s, None, inner_s)
            self.add(inner_e, None, e)
            if isinstance(expr, (Star, Opt)):
                self.add(s, None, e)
            if isinstance(expr, (Star, Plus)):
                self.add(inner_e, None, inner_s)
            return s, e
        raise TypeError(f"unhandled grammar node: {type(expr).__name__}")


def compile_rule(expr: Expr) -> RuleNFA:
    b = _Builder()
    start, accept = b.build(expr)
    return RuleNFA(start=start, accept=accept, num_states=b._n, edges=b.edges)


def compile_rules(grammar: Grammar) -> dict[str, RuleNFA]:
    return {name: compile_rule(expr) for name, expr in grammar.rules.items()}
