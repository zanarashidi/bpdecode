"""CPU reference constraint -- the correctness oracle for all later backends.

``RegexConstraint`` wraps a :class:`~bpdecode.automaton.TokenDFA` and tracks a
single DFA state.  It is intentionally simple and un-optimised: every other
implementation (batched CPU, CUDA) is validated by differential testing
against this one.
"""

from __future__ import annotations

from .automaton import DEAD, TokenDFA
from .interface import BaseConstraint
from .regex import compile_regex
from .tokenizer import Vocabulary


class RegexConstraint(BaseConstraint):
    def __init__(self, pattern: str, vocab: Vocabulary) -> None:
        self._pattern = pattern
        self._tdfa = TokenDFA(compile_regex(pattern), vocab)
        self._state = self._tdfa.start
        self._history: list[int] = []

    @classmethod
    def from_token_dfa(cls, tdfa: TokenDFA) -> RegexConstraint:
        obj = cls.__new__(cls)
        obj._pattern = "<precompiled>"
        obj._tdfa = tdfa
        obj._state = tdfa.start
        obj._history = []
        return obj

    @property
    def vocab_size(self) -> int:
        return self._tdfa.vocab.size

    @property
    def state(self) -> int:
        return self._state

    def reset(self) -> None:
        self._state = self._tdfa.start
        self._history.clear()

    def accepts(self, token_id: int) -> bool:
        return self._tdfa.step(self._state, token_id) != DEAD

    def advance(self, token_id: int) -> None:
        nxt = self._tdfa.step(self._state, token_id)
        if nxt == DEAD:
            raise ValueError(
                f"token {token_id} is not accepted from state {self._state}"
            )
        self._history.append(token_id)
        eos = self._tdfa.vocab.eos_id
        if token_id != eos:
            self._state = nxt

    def is_complete(self) -> bool:
        return self._tdfa.is_accepting(self._state)

    def allowed_ids(self) -> frozenset[int]:
        return self._tdfa.allowed(self._state)
