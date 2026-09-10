"""PDA / CFGConstraint against an independent recursive grammar matcher.

The matcher shares only the IR with the PDA; it is a from-scratch second
implementation of "is this string in the language", used as the oracle.
Test grammars are non-left-recursive (the PDA and the matcher both cap that).
"""

from __future__ import annotations

import itertools

import pytest

from bpdecode.grammar.constraint import CFGConstraint
from bpdecode.grammar.gbnf import parse_gbnf
from bpdecode.grammar.ir import (
    Alt,
    CharSet,
    Concat,
    Empty,
    Grammar,
    Opt,
    Plus,
    Ref,
    Star,
)
from bpdecode.grammar.pda import PDA, CompiledGrammar
from bpdecode.tokenizer import Vocabulary


def language_member(grammar: Grammar, text: str) -> bool:
    """True iff ``text`` is a complete derivation of ``grammar.root``."""
    rules = grammar.rules
    inprogress: set[tuple[str, int]] = set()

    def match(expr: object, pos: int) -> set[int]:
        if isinstance(expr, Empty):
            return {pos}
        if isinstance(expr, CharSet):
            if pos < len(text):
                cp = ord(text[pos])
                if any(lo <= cp <= hi for lo, hi in expr.ranges):
                    return {pos + 1}
            return set()
        if isinstance(expr, Ref):
            key = (expr.name, pos)
            if key in inprogress:  # left-recursion cycle -> no match on this path
                return set()
            inprogress.add(key)
            try:
                return match(rules[expr.name], pos)
            finally:
                inprogress.discard(key)
        if isinstance(expr, Concat):
            cur = {pos}
            for part in expr.parts:
                nxt: set[int] = set()
                for p in cur:
                    nxt |= match(part, p)
                cur = nxt
                if not cur:
                    break
            return cur
        if isinstance(expr, Alt):
            out: set[int] = set()
            for o in expr.options:
                out |= match(o, pos)
            return out
        if isinstance(expr, Opt):
            return {pos} | match(expr.node, pos)
        if isinstance(expr, (Star, Plus)):
            reached = set() if isinstance(expr, Plus) else {pos}
            frontier = {pos}
            while frontier:
                nf: set[int] = set()
                for p in frontier:
                    for q in match(expr.node, p):
                        if q not in reached:
                            reached.add(q)
                            nf.add(q)
                frontier = nf
            return reached
        raise TypeError(type(expr))

    return len(text) in match(rules[grammar.root], 0)


GRAMMARS = {
    "yesno": 'root ::= "yes" | "no"',
    "nested-parens": 'root ::= "(" root ")" | "x"',
    "csv-ish": 'root ::= field ("," field)*\nfield ::= [a-z]+',
    "json-array": (
        'root  ::= "[" (val ("," val)*)? "]"\n'
        'val   ::= "[" (val ("," val)*)? "]" | [0-9]+'
    ),
    "kv": (
        'root  ::= "{" pair ("," pair)* "}"\n'
        'pair  ::= key ":" [0-9]\n'
        'key   ::= [a-c]'
    ),
    "opt-sign": 'root ::= sign? [0-9]+ frac?\nsign ::= "-"\nfrac ::= "." [0-9]+',
}
ALPHABETS = {
    "yesno": "yesno",
    "nested-parens": "()x",
    "csv-ish": "ab,",
    "json-array": "[]0,",
    "kv": "{}abc:0,",
    "opt-sign": "-.0",
}


@pytest.mark.parametrize("name", list(GRAMMARS))
def test_membership_matches_oracle(name: str) -> None:
    grammar = parse_gbnf(GRAMMARS[name])
    compiled = CompiledGrammar.build(grammar)
    alpha = ALPHABETS[name]

    tested = 0
    for length in range(7):
        for combo in itertools.product(alpha, repeat=length):
            s = "".join(combo)
            want = language_member(grammar, s)
            pda = PDA(compiled)
            ok = all(pda.advance_byte(ord(c)) for c in s) if s else True
            got = ok and not pda.dead() and pda.is_complete()
            assert got == want, (name, repr(s), want, got)
            tested += 1
    assert tested > 50


VOCAB = Vocabulary.from_tokens(
    ["(", ")", "x", "y", "yes", "no", "{", "}", "a", "b", ":", "0", "1", ",", "<eos>"],
    eos_id=14,
)


def _grammar_alphabet(grammar: Grammar) -> str:
    cps: set[int] = set()

    def walk(e: object) -> None:
        if isinstance(e, CharSet):
            for lo, hi in e.ranges:
                cps.update({lo, hi, min(hi, lo + 1)})
        elif isinstance(e, Concat):
            [walk(p) for p in e.parts]
        elif isinstance(e, Alt):
            [walk(o) for o in e.options]
        elif isinstance(e, (Star, Plus, Opt)):
            walk(e.node)

    for expr in grammar.rules.values():
        walk(expr)
    return "".join(chr(c) for c in sorted(cps) if c < 0x110000)


def _is_viable_prefix(grammar: Grammar, s: str, k: int = 4) -> bool:
    """True iff some `s + w` with |w| <= k is in the language."""
    if language_member(grammar, s):
        return True
    alpha = _grammar_alphabet(grammar)
    for n in range(1, k + 1):
        for combo in itertools.product(alpha, repeat=n):
            if language_member(grammar, s + "".join(combo)):
                return True
    return False


@pytest.mark.parametrize(
    "src,prefixes",
    [
        ('root ::= "(" root ")" | "x" | "y"', [(), ("(",), ("(", "("), ("(", "x")]),
        ('root ::= "{" p ("," p)* "}"\np ::= [a-b] ":" [0-1]', [(), ("{",), ("{", "a")]),
        ('root ::= "yes" | "no"', [(), ("y",), ("yes",), ("no",)]),
    ],
)
def test_cfgconstraint_allows_exactly_the_viable_tokens(src: str, prefixes: list) -> None:
    grammar = parse_gbnf(src)
    for prefix in prefixes:
        con = CFGConstraint(grammar, VOCAB)
        text = ""
        for tok in prefix:
            con.advance(VOCAB.token_bytes.index(tok.encode()))
            text += tok

        got = {t for t in range(VOCAB.size) if con.accepts(t)}
        expected = set()
        for t in range(VOCAB.size):
            if t == VOCAB.eos_id:
                if language_member(grammar, text):
                    expected.add(t)
            elif _is_viable_prefix(grammar, text + VOCAB.token_bytes[t].decode("latin1")):
                expected.add(t)
        assert got == expected, (src, prefix, sorted(got), sorted(expected))


def test_cfgconstraint_advance_and_complete() -> None:
    con = CFGConstraint('root ::= "(" root ")" | "x"', VOCAB)
    for tok in ["(", "(", "x", ")", ")"]:
        con.advance(VOCAB.token_bytes.index(tok.encode()))
    assert con.is_complete()
    assert con.accepts(VOCAB.eos_id)
    with pytest.raises(ValueError):
        con.advance(VOCAB.token_bytes.index(b")"))


def test_rejects_invalid_token() -> None:
    con = CFGConstraint('root ::= "yes" | "no"', VOCAB)
    with pytest.raises(ValueError):
        con.advance(VOCAB.token_bytes.index(b"x"))
