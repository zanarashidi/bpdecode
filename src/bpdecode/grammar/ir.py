"""Grammar intermediate representation.

A :class:`Grammar` is a set of named rules, each an expression tree.  The
expression nodes are the regex AST (:mod:`bpdecode.regex.parser`) plus
:class:`Ref` -- a reference to another rule.  That reuse means a rule with no
``Ref`` compiles through the exact same NFA -> byte-DFA path as a regex; a rule
*with* ``Ref`` needs the pushdown machinery in :mod:`bpdecode.grammar.pda`.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..regex.parser import Alt, CharSet, Concat, Empty, Opt, Plus, Star

__all__ = [
    "Alt",
    "CharSet",
    "Concat",
    "Empty",
    "Grammar",
    "Opt",
    "Plus",
    "Ref",
    "Star",
    "Expr",
]


@dataclass(frozen=True)
class Ref:
    """A reference to another rule by name."""

    name: str


Expr = Empty | CharSet | Concat | Alt | Star | Plus | Opt | Ref


@dataclass(frozen=True)
class Grammar:
    """Named rules plus the entry rule.

    ``rules`` maps a rule name to its expression tree.  ``root`` names the rule a
    complete derivation must match.
    """

    rules: dict[str, Expr]
    root: str = "root"

    def __post_init__(self) -> None:
        if self.root not in self.rules:
            raise KeyError(f"root rule {self.root!r} is not defined")
        missing = self._undefined_refs()
        if missing:
            raise KeyError(f"undefined rule(s): {', '.join(sorted(missing))}")

    def _undefined_refs(self) -> set[str]:
        seen: set[str] = set()

        def walk(e: Expr) -> None:
            if isinstance(e, Ref):
                if e.name not in self.rules:
                    seen.add(e.name)
            elif isinstance(e, Concat):
                for p in e.parts:
                    walk(p)
            elif isinstance(e, Alt):
                for o in e.options:
                    walk(o)
            elif isinstance(e, (Star, Plus, Opt)):
                walk(e.node)

        for expr in self.rules.values():
            walk(expr)
        return seen

    def is_regular(self) -> bool:
        """True when no rule (transitively) references another -- a plain regex."""
        return not any(_has_ref(e) for e in self.rules.values())


def _has_ref(e: Expr) -> bool:
    if isinstance(e, Ref):
        return True
    if isinstance(e, Concat):
        return any(_has_ref(p) for p in e.parts)
    if isinstance(e, Alt):
        return any(_has_ref(o) for o in e.options)
    if isinstance(e, (Star, Plus, Opt)):
        return _has_ref(e.node)
    return False
