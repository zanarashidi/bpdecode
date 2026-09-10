"""CPU reference constraint for context-free grammars.

``CFGConstraint`` is to CFGs what
:class:`~bpdecode.reference.RegexConstraint` is to regexes: a small, un-optimised
oracle.  It tracks a :class:`~bpdecode.grammar.pda.PDA` and answers ``accepts`` /
``advance`` / ``allowed_ids`` by simulating a token's bytes through the automaton.

If the grammar is regular (no rule references anything), compile it to a
:class:`~bpdecode.regex.compile.DFA` instead -- this class still works but the
DFA path is far faster.
"""

from __future__ import annotations

from ..interface import BaseConstraint
from ..tokenizer import Vocabulary
from .gbnf import parse_gbnf
from .ir import Grammar
from .pda import PDA, CompiledGrammar


class CFGConstraint(BaseConstraint):
    def __init__(
        self,
        grammar: str | Grammar,
        vocab: Vocabulary,
        *,
        root: str = "root",
        max_depth: int = 64,
    ) -> None:
        g = grammar if isinstance(grammar, Grammar) else parse_gbnf(grammar, root)
        self._compiled = CompiledGrammar.build(g)
        self._vocab = vocab
        self._max_depth = max_depth
        self._pda = PDA(self._compiled, max_depth=max_depth)
        self._token_bytes: list[bytes] = list(vocab.token_bytes)
        self._history: list[int] = []

    @classmethod
    def from_compiled(
        cls, compiled: CompiledGrammar, vocab: Vocabulary, *, max_depth: int = 64
    ) -> CFGConstraint:
        obj = cls.__new__(cls)
        obj._compiled = compiled
        obj._vocab = vocab
        obj._max_depth = max_depth
        obj._pda = PDA(compiled, max_depth=max_depth)
        obj._token_bytes = list(vocab.token_bytes)
        obj._history = []
        return obj

    @property
    def vocab_size(self) -> int:
        return self._vocab.size

    def reset(self) -> None:
        self._pda = PDA(self._compiled, max_depth=self._max_depth)
        self._history.clear()

    def is_complete(self) -> bool:
        return self._pda.is_complete()

    def _feeds_ok(self, data: bytes) -> bool:
        snapshot = self._pda.configs
        try:
            for b in data:
                if not self._pda.advance_byte(b):
                    return False
            return not self._pda.dead()
        finally:
            self._pda.set_configs(snapshot)

    def accepts(self, token_id: int) -> bool:
        if token_id == self._vocab.eos_id:
            return self.is_complete()
        return self._feeds_ok(self._token_bytes[token_id])

    def advance(self, token_id: int) -> None:
        if token_id == self._vocab.eos_id:
            if not self.is_complete():
                raise ValueError("EOS is not valid: the string is not a complete match")
            self._history.append(token_id)
            return
        for b in self._token_bytes[token_id]:
            if not self._pda.advance_byte(b):
                raise ValueError(
                    f"token {token_id} is not accepted from the current state"
                )
        self._history.append(token_id)

    def allowed_ids(self) -> frozenset[int]:
        ids = {t for t in range(self._vocab.size) if self.accepts(t)}
        if self._vocab.eos_id is not None and self.is_complete():
            ids.add(self._vocab.eos_id)
        return frozenset(ids)
