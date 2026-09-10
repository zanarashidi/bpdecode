"""GBNF parsing -> grammar IR."""

from __future__ import annotations

import pytest

from bpdecode.grammar import GBNFSyntaxError, parse_gbnf
from bpdecode.grammar.ir import Alt, CharSet, Concat, Opt, Plus, Ref, Star


def test_simple_alternation() -> None:
    g = parse_gbnf('root ::= "yes" | "no"')
    assert g.root == "root"
    assert isinstance(g.rules["root"], Alt)
    assert g.is_regular()


def test_rule_reference_and_regularity() -> None:
    g = parse_gbnf(
        """
        root  ::= obj
        obj   ::= "{" pair "}"
        pair  ::= name ":" name
        name  ::= [a-z]+
        """
    )
    assert not g.is_regular()
    assert isinstance(g.rules["obj"], Concat)
    assert any(isinstance(p, Ref) for p in g.rules["obj"].parts)


def test_repetition_sugar_expands() -> None:
    g = parse_gbnf('root ::= "a"{2,4}')
    node = g.rules["root"]
    assert isinstance(node, Concat)
    kinds = [type(p) for p in node.parts]
    assert kinds == [CharSet, CharSet, Opt, Opt]

    g2 = parse_gbnf('root ::= "x"{2,}')
    assert isinstance(g2.rules["root"], Concat)
    assert isinstance(g2.rules["root"].parts[-1], Star)

    assert isinstance(parse_gbnf('root ::= "x"{3}').rules["root"], Concat)


def test_char_class_negation_and_escapes() -> None:
    g = parse_gbnf(r'root ::= [^"\\] | [\x41-\x5A]')
    assert isinstance(g.rules["root"], Alt)
    cls = g.rules["root"].options[1]
    assert isinstance(cls, CharSet)
    assert cls.ranges == ((0x41, 0x5A),)


def test_quantifiers_and_groups() -> None:
    g = parse_gbnf('root ::= ("ab" | "c")* [0-9]+ "z"?')
    parts = g.rules["root"].parts
    assert isinstance(parts[0], Star)
    assert isinstance(parts[1], Plus)
    assert isinstance(parts[2], Opt)


def test_comments_and_whitespace() -> None:
    g = parse_gbnf(
        """
        # the entry point
        root ::= greeting "!"   # trailing comment
        greeting ::= "hi" | "yo"
        """
    )
    assert isinstance(g.rules["root"], Concat)


@pytest.mark.parametrize(
    "src",
    [
        'root := "x"',
        'root ::= "unterminated',
        "root ::= [a-",
        "root ::= undefined_rule",
        'root ::= "a"{4,2}',
        'notroot ::= "x"',  # no root
        "",
    ],
)
def test_syntax_errors(src: str) -> None:
    with pytest.raises((GBNFSyntaxError, KeyError)):
        parse_gbnf(src)


def test_empty_rhs_is_allowed() -> None:
    g = parse_gbnf('root ::= "a" opt\nopt ::=')
    assert g.rules["opt"].__class__.__name__ == "Empty"
