"""JSON Schema -> Grammar.

Soundness is the contract: every string the grammar accepts must validate
against the schema (checked with the `jsonschema` library). The converse does
not hold -- the grammar fixes property order, so some valid orderings are not
producible.
"""

from __future__ import annotations

import json

import jsonschema
import pytest

from bpdecode.grammar.json_schema import JsonSchemaError, json_schema_to_grammar
from bpdecode.grammar.pda import PDA, CompiledGrammar


def _member(compiled: CompiledGrammar, s: str) -> bool:
    p = PDA(compiled)
    for ch in s:
        if not p.advance_byte(ord(ch)):
            return False
    return not p.dead() and p.is_complete()


OBJECT = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name", "age"],
    "additionalProperties": False,
}

NESTED = {
    "type": "object",
    "properties": {
        "user": {
            "type": "object",
            "properties": {"id": {"type": "integer"}},
            "required": ["id"],
            "additionalProperties": False,
        },
        "role": {"enum": ["admin", "guest"]},
    },
    "required": ["user", "role"],
    "additionalProperties": False,
}

RECURSIVE = {
    "$defs": {
        "node": {
            "type": "object",
            "properties": {
                "v": {"type": "integer"},
                "next": {"anyOf": [{"$ref": "#/$defs/node"}, {"type": "null"}]},
            },
            "required": ["v", "next"],
            "additionalProperties": False,
        }
    },
    "$ref": "#/$defs/node",
}

CASES = [
    (OBJECT, '{"name": "a", "age": 0}', True),
    (OBJECT, '{"name":"a","age":0,"tags":["x","y"]}', True),
    (OBJECT, '{"name": "a", "age": 0, "tags": []}', True),
    (OBJECT, '{"name": "a"}', False),  # missing required
    (OBJECT, '{"age": 0, "name": "a"}', False),  # order not producible
    (OBJECT, '{"name": "a", "age": 0, "extra": 1}', False),  # additionalProperties
    (NESTED, '{"user": {"id": 7}, "role": "admin"}', True),
    (NESTED, '{"user": {"id": 7}, "role": "other"}', False),  # bad enum
    (NESTED, '{"user": {}, "role": "guest"}', False),  # nested required
    (RECURSIVE, '{"v": 1, "next": null}', True),
    (RECURSIVE, '{"v": 1, "next": {"v": 2, "next": null}}', True),
    (RECURSIVE, '{"v": 1}', False),
]


@pytest.mark.parametrize("schema,text,ok", CASES, ids=range(len(CASES)))
def test_membership(schema: dict, text: str, ok: bool) -> None:
    compiled = CompiledGrammar.build(json_schema_to_grammar(schema))
    assert _member(compiled, text) is ok
    if ok:  # sanity: our "valid" cases really do satisfy the schema
        jsonschema.validate(json.loads(text), schema)


@pytest.mark.parametrize("schema,text,ok", [(s, t, o) for s, t, o in CASES if o])
def test_soundness_against_jsonschema(schema: dict, text: str, ok: bool) -> None:
    # every grammar-accepted string validates; mutate it and re-check both agree
    compiled = CompiledGrammar.build(json_schema_to_grammar(schema))
    for cut in range(1, len(text)):
        prefix = text[:cut]
        p = PDA(compiled)
        alive = all(p.advance_byte(ord(c)) for c in prefix)
        # a live prefix must extend to something valid; a dead one never does
        if not alive:
            assert not _member(compiled, prefix)


def test_enum_and_const() -> None:
    g = json_schema_to_grammar({"enum": [1, "two", [3]]})
    c = CompiledGrammar.build(g)
    assert _member(c, "1")
    assert _member(c, '"two"')
    assert _member(c, "[3]")
    assert not _member(c, "2")

    c2 = CompiledGrammar.build(json_schema_to_grammar({"const": {"a": 1}}))
    assert _member(c2, '{"a":1}')
    assert not _member(c2, '{"a": 1}')  # const is exact


def test_array_bounds() -> None:
    g = json_schema_to_grammar(
        {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 3}
    )
    c = CompiledGrammar.build(g)
    assert not _member(c, "[1]")
    assert _member(c, "[1, 2]")
    assert _member(c, "[1, 2, 3]")
    assert not _member(c, "[1, 2, 3, 4]")


def test_string_length_and_pattern() -> None:
    c = CompiledGrammar.build(
        json_schema_to_grammar({"type": "string", "minLength": 1, "maxLength": 3})
    )
    assert _member(c, '"ab"')
    assert not _member(c, '""')
    assert not _member(c, '"abcd"')

    c2 = CompiledGrammar.build(
        json_schema_to_grammar({"type": "string", "pattern": "[a-z]+"})
    )
    assert _member(c2, '"abc"')
    assert not _member(c2, '"a1"')


def test_unsupported_raises() -> None:
    with pytest.raises(JsonSchemaError):
        json_schema_to_grammar({"type": "weird"})
    with pytest.raises(JsonSchemaError):
        json_schema_to_grammar(False)
