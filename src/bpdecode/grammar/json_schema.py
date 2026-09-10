"""Compile a subset of JSON Schema to a :class:`~bpdecode.grammar.ir.Grammar`.

Supported: ``type`` (object / array / string / number / integer / boolean /
null), ``properties`` + ``required`` + ``additionalProperties``, ``items`` +
``minItems`` / ``maxItems``, ``enum``, ``const``, string ``pattern`` /
``minLength`` / ``maxLength``, ``anyOf`` / ``oneOf``, and local ``$ref`` into
``$defs`` / ``definitions``.

Objects are emitted in **schema property order** -- every string produced
validates, but not every valid property ordering is producible (the standard
soundness-over-completeness tradeoff, as in llama.cpp's grammar converter).
Numeric ``minimum`` / ``maximum`` are ignored (not expressible as a grammar).
"""

from __future__ import annotations

import json
from typing import Any

from ..regex.parser import parse as parse_regex
from .ir import Alt, CharSet, Concat, Empty, Expr, Grammar, Opt, Ref, Star

__all__ = ["json_schema_to_grammar", "JsonSchemaError"]


class JsonSchemaError(ValueError):
    """Unsupported or malformed schema."""


def _lit(text: str) -> Expr:
    parts = [CharSet(((ord(c), ord(c)),)) for c in text]
    if not parts:
        return Empty()
    return parts[0] if len(parts) == 1 else Concat(tuple(parts))


def _json_literal(value: Any) -> Expr:
    """A grammar matching exactly ``json.dumps(value)`` (compact)."""
    return _lit(json.dumps(value, separators=(",", ":")))


_PRIMITIVES = r"""
ws        ::= [ \t\n\r]*
json-value ::= json-object | json-array | json-string | json-number | "true" | "false" | "null"
json-object ::= "{" ws (json-member (ws "," ws json-member)*)? ws "}"
json-member ::= json-string ws ":" ws json-value
json-array ::= "[" ws (json-value (ws "," ws json-value)*)? ws "]"
json-string ::= "\"" json-char* "\""
json-char  ::= [^"\\\x00-\x1F] | "\\" (["\\/bfnrt] | "u" hex hex hex hex)
hex        ::= [0-9a-fA-F]
json-number ::= "-"? json-int json-frac? json-exp?
json-int   ::= "0" | [1-9] [0-9]*
json-frac  ::= "." [0-9]+
json-exp   ::= [eE] [+-]? [0-9]+
"""


class _Builder:
    def __init__(self, defs: dict[str, Any]) -> None:
        self.rules: dict[str, Expr] = {}
        self.defs = defs
        self._n = 0
        self._ref_rule: dict[str, str] = {}

    def emit(self, name: str, expr: Expr) -> str:
        self.rules[name] = expr
        return name

    # --- schema dispatch -------------------------------------------
    def compile(self, schema: Any) -> Expr:
        if schema is True or schema == {}:
            return Ref("json-value")
        if schema is False:
            raise JsonSchemaError("schema `false` matches nothing")
        if not isinstance(schema, dict):
            raise JsonSchemaError(f"schema must be an object, got {type(schema).__name__}")

        if "$ref" in schema:
            return Ref(self._resolve_ref(schema["$ref"]))
        if "const" in schema:
            return _json_literal(schema["const"])
        if "enum" in schema:
            return Alt(tuple(_json_literal(v) for v in schema["enum"]))
        for key in ("anyOf", "oneOf"):
            if key in schema:
                return Alt(tuple(self.compile(s) for s in schema[key]))

        t = schema.get("type")
        if isinstance(t, list):
            return Alt(tuple(self.compile({**schema, "type": one}) for one in t))
        if t is None:
            return Ref("json-value")
        handler = {
            "object": self._object,
            "array": self._array,
            "string": self._string,
            "number": lambda s: Ref("json-number"),
            "integer": self._integer,
            "boolean": lambda s: Alt((_lit("true"), _lit("false"))),
            "null": lambda s: _lit("null"),
        }.get(t)
        if handler is None:
            raise JsonSchemaError(f"unsupported type {t!r}")
        return handler(schema)

    def _resolve_ref(self, ref: str) -> str:
        if ref in self._ref_rule:
            return self._ref_rule[ref]
        for prefix in ("#/$defs/", "#/definitions/"):
            if ref.startswith(prefix):
                key = ref[len(prefix) :]
                if key not in self.defs:
                    raise JsonSchemaError(f"$ref target not found: {ref}")
                name = f"_def_{key}"
                self._ref_rule[ref] = name
                self.rules[name] = Empty()  # placeholder to break cycles
                self.rules[name] = self.compile(self.defs[key])
                return name
        raise JsonSchemaError(f"unsupported $ref: {ref}")

    # --- composite types ------------------------------------------
    def _object(self, schema: dict) -> Expr:
        props: dict[str, Any] = schema.get("properties", {})
        required = set(schema.get("required", []))
        additional = schema.get("additionalProperties", True)
        items = list(props.items())

        def rest(i: int, seen: bool) -> Expr:
            if i == len(items):
                return self._extra_members(additional, seen)
            name, subschema = items[i]
            kv = Concat(
                (_json_literal(name), Ref("ws"), _lit(":"), Ref("ws"), self.compile(subschema))
            )
            with_sep = kv if not seen else Concat((Ref("ws"), _lit(","), Ref("ws"), kv))
            include = Concat((with_sep, rest(i + 1, True)))
            if name in required:
                return include
            return Alt((rest(i + 1, seen), include))

        body = rest(0, False)
        return Concat((_lit("{"), Ref("ws"), body, Ref("ws"), _lit("}")))

    def _extra_members(self, additional: Any, seen: bool) -> Expr:
        if additional is False:
            return Empty()
        value = Ref("json-value") if additional is True else self.compile(additional)
        member = Concat((Ref("json-string"), Ref("ws"), _lit(":"), Ref("ws"), value))
        sep_member = Concat((Ref("ws"), _lit(","), Ref("ws"), member))
        if seen:
            return Star(sep_member)
        return Opt(Concat((member, Star(sep_member))))

    def _array(self, schema: dict) -> Expr:
        item = self.compile(schema.get("items", True))
        lo = int(schema.get("minItems", 0))
        hi = schema.get("maxItems")
        sep_item = Concat((Ref("ws"), _lit(","), Ref("ws"), item))

        if hi is None:
            tail = Star(sep_item) if lo <= 1 else Concat(
                (*([sep_item] * (lo - 1)), Star(sep_item))
            )
            body: Expr = Empty() if lo == 0 else Concat((item, tail))
            if lo == 0:
                body = Opt(Concat((item, Star(sep_item))))
        else:
            hi = int(hi)
            if hi < lo:
                raise JsonSchemaError("maxItems < minItems")
            opts: list[Expr] = []
            for count in range(lo, hi + 1):
                if count == 0:
                    opts.append(Empty())
                else:
                    opts.append(Concat((item, *([sep_item] * (count - 1)))))
            body = opts[0] if len(opts) == 1 else Alt(tuple(opts))
        return Concat((_lit("["), Ref("ws"), body, Ref("ws"), _lit("]")))

    def _string(self, schema: dict) -> Expr:
        if "pattern" in schema:
            inner = _wrap_regex(parse_regex(schema["pattern"]))
            return Concat((_lit('"'), inner, _lit('"')))
        lo = schema.get("minLength")
        hi = schema.get("maxLength")
        if lo is None and hi is None:
            return Ref("json-string")
        lo = int(lo or 0)
        char = Ref("json-char")
        parts: list[Expr] = [char] * lo
        if hi is None:
            parts.append(Star(char))
        else:
            parts.extend(Opt(char) for _ in range(int(hi) - lo))
        body: Expr = Empty() if not parts else (
            parts[0] if len(parts) == 1 else Concat(tuple(parts))
        )
        return Concat((_lit('"'), body, _lit('"')))

    def _integer(self, schema: dict) -> Expr:
        return Concat((Opt(_lit("-")), Ref("json-int")))


def _wrap_regex(node: Any) -> Expr:
    """The regex AST already shares node types with the grammar IR."""
    return node if node is not None else Empty()


def json_schema_to_grammar(schema: Any, *, root: str = "root") -> Grammar:
    """Compile ``schema`` (a parsed JSON Schema) into a :class:`Grammar`."""
    from .gbnf import parse_gbnf

    primitives = parse_gbnf(_PRIMITIVES + '\nroot ::= "x"').rules
    del primitives["root"]

    defs = {}
    if isinstance(schema, dict):
        defs = schema.get("$defs") or schema.get("definitions") or {}

    b = _Builder(defs)
    b.rules.update(primitives)
    b.emit(root, b.compile(schema))
    return Grammar(rules=b.rules, root=root)
