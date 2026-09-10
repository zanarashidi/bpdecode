"""Token-level automaton: lift a code-point :class:`DFA` to operate on token ids.

Given a code-point DFA and a :class:`Vocabulary`, :class:`TokenDFA` answers the
two questions the decoding loop needs at every step:

* ``allowed(state)`` -- the set of token ids that keep a valid path alive
* ``step(state, token_id)`` -- the DFA state after emitting that token

Transitions are computed lazily and memoised, mirroring the adaptive mask cache
the GPU kernels will use later.  A token is *allowed* from ``state`` when
feeding its bytes through the DFA never hits the dead state and the resulting
state can still reach an accepting state (``DFA.live_states``).  The EOS token
is allowed exactly when ``state`` is accepting.
"""

from __future__ import annotations

from .regex.compile import DFA
from .tokenizer import Vocabulary

DEAD = -1


class TokenDFA:
    def __init__(self, dfa: DFA, vocab: Vocabulary) -> None:
        self.dfa = dfa
        self.vocab = vocab
        self._live = dfa.live_states()
        # token id -> the raw bytes it contributes (the DFA alphabet)
        self._token_bytes: list[tuple[int, ...]] = [
            tuple(b) for b in vocab.token_bytes
        ]
        self._step_cache: dict[tuple[int, int], int] = {}
        self._allowed_cache: dict[int, frozenset[int]] = {}

    @property
    def start(self) -> int:
        return self.dfa.start

    def is_accepting(self, state: int) -> bool:
        return state in self.dfa.accept

    def _run(self, state: int, data: tuple[int, ...]) -> int:
        cur = state
        for byte in data:
            cur = self.dfa.step(cur, byte)
            if cur == self.dfa.dead:
                return DEAD
        return cur

    def step(self, state: int, token_id: int) -> int:
        """DFA state after emitting ``token_id``; ``DEAD`` (-1) if it is invalid."""
        if token_id == self.vocab.eos_id:
            return state if self.is_accepting(state) else DEAD
        key = (state, token_id)
        hit = self._step_cache.get(key)
        if hit is not None:
            return hit
        nxt = self._run(state, self._token_bytes[token_id])
        if nxt != DEAD and nxt not in self._live:
            nxt = DEAD
        self._step_cache[key] = nxt
        return nxt

    def allowed(self, state: int) -> frozenset[int]:
        """Every token id (EOS included) that is valid from ``state``."""
        hit = self._allowed_cache.get(state)
        if hit is not None:
            return hit
        ids = {tid for tid in range(self.vocab.size) if self.step(state, tid) != DEAD}
        if self.vocab.eos_id is not None and self.is_accepting(state):
            ids.add(self.vocab.eos_id)
        frozen = frozenset(ids)
        self._allowed_cache[state] = frozen
        return frozen

    def mask(self, state: int, out: list[bool] | None = None) -> list[bool]:
        """Boolean allow-mask of length ``vocab.size`` for ``state``."""
        allowed = self.allowed(state)
        if out is None:
            out = [False] * self.vocab.size
        for i in range(self.vocab.size):
            out[i] = i in allowed
        return out
