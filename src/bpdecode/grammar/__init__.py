"""Context-free grammar constrained decoding.

* :mod:`~bpdecode.grammar.gbnf` -- parse GBNF source to a :class:`Grammar`
* :mod:`~bpdecode.grammar.ir` -- the grammar IR (regex AST + :class:`Ref`)
"""

from __future__ import annotations

from .constraint import CFGConstraint
from .gbnf import GBNFSyntaxError, parse_gbnf
from .ir import Grammar, Ref
from .json_schema import JsonSchemaError, json_schema_to_grammar
from .pda import PDA, CompiledGrammar, PDAOverflow

__all__ = [
    "CFGConstraint",
    "CompiledGrammar",
    "GBNFSyntaxError",
    "Grammar",
    "JsonSchemaError",
    "PDA",
    "PDAOverflow",
    "Ref",
    "json_schema_to_grammar",
    "parse_gbnf",
]
