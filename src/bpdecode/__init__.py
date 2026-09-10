"""bpdecode -- GPU-accelerated constrained decoding for LLM inference.

Phase 0 ships the host-side scaffolding only: a regex -> DFA front-end, a
token-level automaton, the :class:`Constraint` interface, and a CPU reference
implementation used as the correctness oracle for the CUDA kernels that land
in later phases.  See ``docs/PLAN.md`` for the roadmap.
"""

from __future__ import annotations

from .automaton import TokenDFA
from .fsa import FsaTable, TokenSymbols, build_reachability, fsa_from_dfa, token_symbols
from .interface import BaseConstraint, Constraint
from .reference import RegexConstraint
from .regex import DFA, RegexSyntaxError, compile_regex
from .tokenizer import DEFAULT_MODEL, Vocabulary

__version__ = "0.0.0"

__all__ = [
    "BaseConstraint",
    "Constraint",
    "DEFAULT_MODEL",
    "DFA",
    "FsaTable",
    "RegexConstraint",
    "RegexSyntaxError",
    "TokenDFA",
    "TokenSymbols",
    "Vocabulary",
    "build_reachability",
    "compile_regex",
    "fsa_from_dfa",
    "token_symbols",
    "__version__",
]
