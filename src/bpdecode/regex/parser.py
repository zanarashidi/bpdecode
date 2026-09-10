"""A small regex parser.

Supported syntax:

    literal characters, with ``\\`` escapes
    ``.``            any Unicode scalar (lowered to its UTF-8 bytes downstream)
    ``*`` ``+`` ``?``  greedy quantifiers (greediness is irrelevant for
                       language membership, which is all we use)
    ``a|b``          alternation
    ``(...)``        grouping
    ``[abc]`` ``[a-z]`` ``[^...]``  character classes

Not yet supported: counted repetition ``{m,n}``, anchors, backreferences,
lookaround, named groups. These raise :class:`RegexSyntaxError`.
"""

from __future__ import annotations

from dataclasses import dataclass

UNICODE_MAX = 0x10FFFF  # AST ranges are code points; compile.py lowers them to UTF-8


class RegexSyntaxError(ValueError):
    """Raised for regex constructs the Phase 0 parser does not implement."""


# --- AST -------------------------------------------------------------------

@dataclass
class Empty:
    """Matches the empty string."""


@dataclass
class CharSet:
    """A set of accepted code points, stored as sorted inclusive ranges."""

    ranges: tuple[tuple[int, int], ...]

    def matches(self, cp: int) -> bool:
        return any(lo <= cp <= hi for lo, hi in self.ranges)


@dataclass
class Concat:
    parts: tuple[Node, ...]


@dataclass
class Alt:
    options: tuple[Node, ...]


@dataclass
class Star:
    node: Node


@dataclass
class Plus:
    node: Node


@dataclass
class Opt:
    node: Node


Node = Empty | CharSet | Concat | Alt | Star | Plus | Opt


# --- parser --------------------------------------------------------------

_CLASS_SHORTHANDS: dict[str, tuple[tuple[int, int], ...]] = {
    "d": ((0x30, 0x39),),
    "w": ((0x30, 0x39), (0x41, 0x5A), (0x5F, 0x5F), (0x61, 0x7A)),
    "s": ((0x09, 0x0D), (0x20, 0x20)),
}


@dataclass
class _Parser:
    src: str
    pos: int = 0

    def peek(self) -> str | None:
        return self.src[self.pos] if self.pos < len(self.src) else None

    def next(self) -> str:
        ch = self.src[self.pos]
        self.pos += 1
        return ch

    def eof(self) -> bool:
        return self.pos >= len(self.src)

    # alternation := concat ('|' concat)*
    def parse_alt(self) -> Node:
        options = [self.parse_concat()]
        while self.peek() == "|":
            self.next()
            options.append(self.parse_concat())
        return options[0] if len(options) == 1 else Alt(tuple(options))

    # concat := repeat*
    def parse_concat(self) -> Node:
        parts: list[Node] = []
        while not self.eof() and self.peek() not in ("|", ")"):
            parts.append(self.parse_repeat())
        if not parts:
            return Empty()
        return parts[0] if len(parts) == 1 else Concat(tuple(parts))

    # repeat := atom ('*' | '+' | '?')*
    def parse_repeat(self) -> Node:
        node = self.parse_atom()
        while (ch := self.peek()) in ("*", "+", "?"):
            self.next()
            node = {"*": Star, "+": Plus, "?": Opt}[ch](node)
        if self.peek() == "{":
            raise RegexSyntaxError("counted repetition {m,n} is not supported yet")
        return node

    def parse_atom(self) -> Node:
        ch = self.peek()
        if ch is None:
            return Empty()
        if ch == "(":
            self.next()
            inner = self.parse_alt()
            if self.peek() != ")":
                raise RegexSyntaxError("unbalanced '('")
            self.next()
            return inner
        if ch == "[":
            return self._parse_class()
        if ch == ".":
            self.next()
            return CharSet(((0, UNICODE_MAX),))
        if ch in ("*", "+", "?"):
            raise RegexSyntaxError(f"nothing to repeat before '{ch}'")
        if ch in ("^", "$"):
            raise RegexSyntaxError("anchors are not supported yet")
        if ch == ")":
            raise RegexSyntaxError("unbalanced ')'")
        if ch == "\\":
            self.next()
            return self._parse_escape()
        self.next()
        cp = ord(ch)
        return CharSet(((cp, cp),))

    def _parse_escape(self) -> CharSet:
        if self.eof():
            raise RegexSyntaxError("trailing backslash")
        esc = self.next()
        if esc in _CLASS_SHORTHANDS:
            return CharSet(_CLASS_SHORTHANDS[esc])
        if esc in ("D", "W", "S"):
            return CharSet(_negate(_CLASS_SHORTHANDS[esc.lower()]))
        mapped = {"n": "\n", "t": "\t", "r": "\r", "f": "\f", "v": "\v"}.get(esc, esc)
        cp = ord(mapped)
        return CharSet(((cp, cp),))

    def _parse_class(self) -> CharSet:
        assert self.next() == "["
        negated = False
        if self.peek() == "^":
            self.next()
            negated = True
        ranges: list[tuple[int, int]] = []
        first = True
        while True:
            ch = self.peek()
            if ch is None:
                raise RegexSyntaxError("unterminated character class")
            if ch == "]" and not first:
                self.next()
                break
            first = False
            lo = self._class_char()
            if self.peek() == "-" and self.src[self.pos + 1 : self.pos + 2] not in ("]", ""):
                self.next()
                hi = self._class_char()
                if hi < lo:
                    raise RegexSyntaxError("character class range is out of order")
                ranges.append((lo, hi))
            else:
                ranges.append((lo, lo))
        merged = _merge(ranges)
        return CharSet(_negate(merged) if negated else merged)

    def _class_char(self) -> int:
        ch = self.next()
        if ch == "\\":
            esc = self.next()
            mapped = {"n": "\n", "t": "\t", "r": "\r", "f": "\f", "v": "\v"}.get(esc, esc)
            return ord(mapped)
        return ord(ch)


def _merge(ranges: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    if not ranges:
        return ()
    ordered = sorted(ranges)
    out = [ordered[0]]
    for lo, hi in ordered[1:]:
        plo, phi = out[-1]
        if lo <= phi + 1:
            out[-1] = (plo, max(phi, hi))
        else:
            out.append((lo, hi))
    return tuple(out)


def _negate(ranges: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    out: list[tuple[int, int]] = []
    cursor = 0
    for lo, hi in sorted(ranges):
        if lo > cursor:
            out.append((cursor, lo - 1))
        cursor = max(cursor, hi + 1)
    if cursor <= UNICODE_MAX:
        out.append((cursor, UNICODE_MAX))
    return tuple(out)


def parse(pattern: str) -> Node:
    """Parse ``pattern`` into an AST, raising :class:`RegexSyntaxError` on failure."""
    p = _Parser(pattern)
    node = p.parse_alt()
    if not p.eof():
        raise RegexSyntaxError(f"unexpected character at position {p.pos}: {p.src[p.pos]!r}")
    return node
