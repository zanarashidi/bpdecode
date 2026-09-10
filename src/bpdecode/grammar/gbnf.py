"""Parse GBNF (llama.cpp grammar format) into a :class:`~bpdecode.grammar.ir.Grammar`.

    root   ::= "yes" | "no"
    ws     ::= [ \\t\\n]*

Supported: ``::=`` rule definitions, ``|`` alternation, juxtaposition = concat,
``"strings"``, ``[a-z]`` / ``[^...]`` char classes, rule references, ``(...)``
grouping, postfix ``* + ?`` and ``{m,n}`` repetition, ``#`` line comments, and
``\\xHH`` / ``\\uHHHH`` escapes.  Code points here; the compiler lowers to UTF-8.
"""

from __future__ import annotations

from .ir import Alt, CharSet, Concat, Empty, Expr, Grammar, Opt, Plus, Ref, Star

_CLASS_ESCAPES = {"n": 0x0A, "t": 0x09, "r": 0x0D, "f": 0x0C, "v": 0x0B, "0": 0x00}
_NAME_START = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
_NAME_REST = _NAME_START + "0123456789-_"


class GBNFSyntaxError(ValueError):
    """Raised for malformed GBNF."""


class _P:
    def __init__(self, src: str) -> None:
        self.s = src
        self.i = 0
        self.n = len(src)

    # --- lexing helpers ------------------------------------------------
    def _skip_ws(self) -> None:
        while self.i < self.n:
            c = self.s[self.i]
            if c in " \t\r\n":
                self.i += 1
            elif c == "#":
                while self.i < self.n and self.s[self.i] != "\n":
                    self.i += 1
            else:
                break

    def eof(self) -> bool:
        self._skip_ws()
        return self.i >= self.n

    def peek(self) -> str:
        self._skip_ws()
        return self.s[self.i] if self.i < self.n else ""

    def error(self, msg: str) -> GBNFSyntaxError:
        return GBNFSyntaxError(f"{msg} at position {self.i}")

    # --- grammar structure ------------------------------------------
    def parse_grammar(self) -> dict[str, Expr]:
        rules: dict[str, Expr] = {}
        while not self.eof():
            name = self._name()
            self._skip_ws()
            if not self.s.startswith("::=", self.i):
                raise self.error(f"expected '::=' after rule {name!r}")
            self.i += 3
            rules[name] = self._alternates()
        if not rules:
            raise self.error("empty grammar")
        return rules

    def _at_rule_start(self) -> bool:
        """True if the next tokens are ``NAME ::=`` -- i.e. a new rule, not a ref."""
        self._skip_ws()
        j = self.i
        if j >= self.n or self.s[j] not in _NAME_START:
            return False
        j += 1
        while j < self.n and self.s[j] in _NAME_REST:
            j += 1
        while j < self.n and self.s[j] in " \t\r\n":
            j += 1
        return self.s.startswith("::=", j)

    def _name(self) -> str:
        self._skip_ws()
        start = self.i
        if self.i >= self.n or self.s[self.i] not in _NAME_START:
            raise self.error("expected a rule name")
        self.i += 1
        while self.i < self.n and self.s[self.i] in _NAME_REST:
            self.i += 1
        return self.s[start : self.i]

    def _alternates(self) -> Expr:
        opts = [self._concat()]
        while self.peek() == "|":
            self.i += 1
            opts.append(self._concat())
        return opts[0] if len(opts) == 1 else Alt(tuple(opts))

    def _concat(self) -> Expr:
        parts: list[Expr] = []
        while True:
            c = self.peek()
            if c in ("", "|", ")"):
                break
            if c in _NAME_START and self._at_rule_start():
                break  # start of the next rule
            parts.append(self._rep())
        if not parts:
            return Empty()
        return parts[0] if len(parts) == 1 else Concat(tuple(parts))

    def _rep(self) -> Expr:
        node = self._atom()
        c = self.peek()
        if c == "*":
            self.i += 1
            return Star(node)
        if c == "+":
            self.i += 1
            return Plus(node)
        if c == "?":
            self.i += 1
            return Opt(node)
        if c == "{":
            return self._counted(node)
        return node

    def _counted(self, node: Expr) -> Expr:
        assert self.s[self.i] == "{"
        self.i += 1
        lo = self._int()
        hi: int | None = lo
        if self.peek() == ",":
            self.i += 1
            self._skip_ws()
            hi = None if self.s[self.i] == "}" else self._int()
        if self.peek() != "}":
            raise self.error("expected '}' to close repetition")
        self.i += 1
        if hi is not None and hi < lo:
            raise self.error("repetition {m,n} has n < m")
        parts: list[Expr] = [node] * lo
        if hi is None:
            parts.append(Star(node))
        else:
            parts.extend(Opt(node) for _ in range(hi - lo))
        if not parts:
            return Empty()
        return parts[0] if len(parts) == 1 else Concat(tuple(parts))

    def _int(self) -> int:
        self._skip_ws()
        start = self.i
        while self.i < self.n and self.s[self.i].isdigit():
            self.i += 1
        if self.i == start:
            raise self.error("expected an integer")
        return int(self.s[start : self.i])

    def _atom(self) -> Expr:
        c = self.peek()
        if c == "(":
            self.i += 1
            inner = self._alternates()
            if self.peek() != ")":
                raise self.error("unbalanced '('")
            self.i += 1
            return inner
        if c == '"':
            return self._string()
        if c == "[":
            return self._charclass()
        if c in _NAME_START:
            return Ref(self._name())
        raise self.error(f"unexpected character {c!r}")

    def _string(self) -> Expr:
        assert self.s[self.i] == '"'
        self.i += 1
        cps: list[int] = []
        while self.i < self.n and self.s[self.i] != '"':
            cps.append(self._char('"'))
        if self.i >= self.n:
            raise self.error("unterminated string")
        self.i += 1  # closing quote
        if not cps:
            return Empty()
        parts = [CharSet(((cp, cp),)) for cp in cps]
        return parts[0] if len(parts) == 1 else Concat(tuple(parts))

    def _charclass(self) -> Expr:
        assert self.s[self.i] == "["
        self.i += 1
        negated = self.i < self.n and self.s[self.i] == "^"
        if negated:
            self.i += 1
        ranges: list[tuple[int, int]] = []
        while self.i < self.n and self.s[self.i] != "]":
            lo = self._char("]")
            if (
                self.i + 1 < self.n
                and self.s[self.i] == "-"
                and self.s[self.i + 1] != "]"
            ):
                self.i += 1
                hi = self._char("]")
                if hi < lo:
                    raise self.error("char class range out of order")
                ranges.append((lo, hi))
            else:
                ranges.append((lo, lo))
        if self.i >= self.n:
            raise self.error("unterminated char class")
        self.i += 1  # closing ]
        merged = _merge(ranges)
        return CharSet(_negate(merged) if negated else merged)

    def _char(self, terminator: str) -> int:
        c = self.s[self.i]
        if c == "\\":
            self.i += 1
            if self.i >= self.n:
                raise self.error("trailing backslash")
            e = self.s[self.i]
            self.i += 1
            if e in _CLASS_ESCAPES:
                return _CLASS_ESCAPES[e]
            if e == "x":
                return self._hex(2)
            if e == "u":
                return self._hex(4)
            if e == "U":
                return self._hex(8)
            return ord(e)  # \\  \"  \]  \-  etc.
        self.i += 1
        return ord(c)

    def _hex(self, n: int) -> int:
        if self.i + n > self.n:
            raise self.error(f"expected {n} hex digits")
        chunk = self.s[self.i : self.i + n]
        try:
            v = int(chunk, 16)
        except ValueError as exc:
            raise self.error(f"bad hex escape {chunk!r}") from exc
        self.i += n
        return v


def _merge(ranges: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    if not ranges:
        return ()
    out = [list(r) for r in sorted(ranges)]
    merged = [out[0]]
    for lo, hi in out[1:]:
        if lo <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return tuple((lo, hi) for lo, hi in merged)


def _negate(ranges: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    out: list[tuple[int, int]] = []
    cursor = 0
    for lo, hi in sorted(ranges):
        if lo > cursor:
            out.append((cursor, lo - 1))
        cursor = max(cursor, hi + 1)
    if cursor <= 0x10FFFF:
        out.append((cursor, 0x10FFFF))
    return tuple(out)


def parse_gbnf(text: str, root: str = "root") -> Grammar:
    """Parse GBNF source into a :class:`Grammar`."""
    return Grammar(rules=_P(text).parse_grammar(), root=root)
