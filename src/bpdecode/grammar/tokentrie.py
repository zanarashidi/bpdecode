"""A byte trie over the vocabulary, for masking against a :class:`PDA`.

Simulating every token's bytes through the PDA independently is O(vocab x
token_len).  Tokens share byte prefixes, so instead we DFS a trie of the vocab
once: feed a byte, recurse, restore the PDA's config-set, and prune the whole
subtree the moment the PDA dies.  Cost is proportional to the number of trie
nodes the grammar keeps alive, which for a tight grammar is a small fraction of
the vocab.
"""

from __future__ import annotations

from ..tokenizer import Vocabulary
from .pda import PDA


class _Node:
    __slots__ = ("children", "token_id")

    def __init__(self) -> None:
        self.children: dict[int, _Node] = {}
        self.token_id: int | None = None  # set if a token ends exactly here


class TokenTrie:
    """Prefix trie of the token byte strings. Build once per vocabulary."""

    def __init__(self, root: _Node, eos_id: int | None) -> None:
        self._root = root
        self._eos_id = eos_id

    @classmethod
    def build(cls, vocab: Vocabulary) -> TokenTrie:
        root = _Node()
        for tid, tb in enumerate(vocab.token_bytes):
            if tid == vocab.eos_id:
                continue
            node = root
            for b in tb:
                node = node.children.setdefault(b, _Node())
            node.token_id = tid
        return cls(root, vocab.eos_id)

    def allowed(self, pda: PDA) -> set[int]:
        """Every token id whose bytes the PDA accepts from its current state,
        plus EOS if the PDA is in a complete state.
        """
        out: set[int] = set()
        snapshot = pda.configs
        self._descend(pda, self._root, out)
        pda.set_configs(snapshot)
        if self._eos_id is not None and pda.is_complete():
            out.add(self._eos_id)
        return out

    def _descend(self, pda: PDA, node: _Node, out: set[int]) -> None:
        if node.token_id is not None:
            out.add(node.token_id)
        if not node.children:
            return
        here = pda.configs
        for b, child in node.children.items():
            if pda.advance_byte(b):
                self._descend(pda, child, out)
            pda.set_configs(here)


_CACHE: dict[int, TokenTrie] = {}


def token_trie(vocab: Vocabulary) -> TokenTrie:
    """Cached trie for ``vocab`` (keyed by identity)."""
    key = id(vocab)
    trie = _CACHE.get(key)
    if trie is None:
        trie = TokenTrie.build(vocab)
        _CACHE[key] = trie
    return trie
