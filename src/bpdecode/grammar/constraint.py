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
from .regular import frame_is_regular, residual_dfa
from .tokentrie import token_trie

_BYTE_SYMS: dict[int, object] = {}


def _byte_symbols(vocab: Vocabulary) -> object:
    """TokenSymbols where each token's symbols are its raw bytes (a 256-way,
    byte-indexed partition). Shared by every residual DFA; built once per vocab.
    """
    from ..fsa import TokenSymbols

    hit = _BYTE_SYMS.get(id(vocab))
    if hit is None:
        offsets = [0]
        total = 0
        for tb in vocab.token_bytes:
            total += len(tb)
            offsets.append(total)
        hit = TokenSymbols(
            vocab_size=vocab.size,
            eos_id=vocab.eos_id if vocab.eos_id is not None else -1,
            offsets=offsets,
            symbols=b"".join(vocab.token_bytes),
        )
        _BYTE_SYMS[id(vocab)] = hit
    return hit


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
        self._token_bytes = vocab.token_bytes
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
        obj._token_bytes = vocab.token_bytes
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
        # the mask is a pure function of (vocab, config-set), shared across every
        # request on this grammar.
        memo = self._compiled.mask_memo
        cfgset = self._pda.configs
        key = (id(self._vocab), cfgset)
        hit = memo.get(key)
        if hit is not None:
            return hit

        tops = {cfg[-1] for cfg in cfgset if cfg}
        c = self._compiled
        # only worth the residual-DFA precompute when many first bytes are
        # allowed (a string / number loop); a narrow frontier means the trie DFS
        # visits few nodes and is cheaper.
        span = sum(hi - lo + 1 for lo, hi in self._pda.first_byte_ranges())
        wide = span >= 32
        if wide and tops and all(
            frame_is_regular(c, c.regular, r, s) for r, s in tops
        ):
            result = self._fast_mask(tops)
        else:
            result = frozenset(token_trie(self._vocab).allowed(self._pda))
        memo[key] = result
        return result

    def _fast_mask(self, tops: set[tuple[str, int]]) -> frozenset[int]:
        """Regular top frames: context-independent tokens come from the dense
        residual-DFA table; only boundary tokens (which could pop the frame)
        are simulated against the real stack.
        """
        allowed: set[int] = set()
        boundary: set[int] = set()
        for r, s in tops:
            ci, bd = self._residual_masks(r, s)
            allowed |= ci
            boundary |= bd
        for t in boundary - allowed:
            if self._feeds_ok(self._token_bytes[t]):
                allowed.add(t)
        if self._vocab.eos_id is not None and self.is_complete():
            allowed.add(self._vocab.eos_id)
        return frozenset(allowed)

    def _residual_masks(
        self, rule: str, state: int
    ) -> tuple[frozenset[int], frozenset[int]]:
        memo = self._compiled.residual_memo
        key = (id(self._vocab), rule, state)
        hit = memo.get(key)
        if hit is not None:
            return hit

        # different NFA states in the same loop yield isomorphic residual DFAs;
        # key the expensive part by the DFA itself (a frozen, hashable dataclass).
        dfa = residual_dfa(self._compiled, rule, state)
        dfa_key = (id(self._vocab), dfa)
        result = memo.get(dfa_key)
        if result is None:
            result = self._masks_for_dfa(dfa)
            memo[dfa_key] = result
        memo[key] = result
        return result

    def _masks_for_dfa(self, dfa: object) -> tuple[frozenset[int], frozenset[int]]:
        import torch

        from ..fsa import fsa_from_dfa
        from ..ops import FsaTensors, build_token_transitions

        fsa = FsaTensors.from_tables(fsa_from_dfa(dfa), _byte_symbols(self._vocab))
        tn = build_token_transitions(fsa, rows=[dfa.start])[0]  # [vocab] int32
        if dfa.accept:
            acc = torch.tensor(sorted(dfa.accept), dtype=tn.dtype)
            ends_accept = torch.isin(tn, acc)
        else:
            ends_accept = torch.zeros_like(tn, dtype=torch.bool)
        ci_mask = (tn != -1) & ~ends_accept
        eos = self._vocab.eos_id
        if eos is not None and eos < ci_mask.shape[0]:
            ci_mask[eos] = False
            ends_accept[eos] = False
        return (
            frozenset(ci_mask.nonzero().flatten().tolist()),
            frozenset(ends_accept.nonzero().flatten().tolist()),
        )
