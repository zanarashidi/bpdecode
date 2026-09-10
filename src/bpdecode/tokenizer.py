"""Vocabulary loading.

A :class:`Vocabulary` is the bridge between token ids and the byte strings the
automaton consumes.  Phase 0 keeps this deliberately small:

* ``from_tokens`` -- build directly from a list of strings (used by tests and
  toy grammars).
* ``from_hf`` -- pull the vocab out of a Hugging Face tokenizer.  Mapping a
  BPE token id to the exact bytes it contributes is genuinely fiddly
  (byte-level BPE, partial UTF-8, added tokens); the implementation here
  handles the common GPT-2 byte-level case and falls back to ``decode`` with a
  logged caveat.  Getting this fully right is tracked for Phase 1.

``from_hf`` defaults to :data:`DEFAULT_MODEL` (``Qwen/Qwen2.5-0.5B``), a small
GPT-2-style byte-level BPE tokenizer that runs comfortably on a laptop.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
"""Default HF model whose tokenizer :meth:`Vocabulary.from_hf` loads."""


@dataclass(frozen=True)
class Vocabulary:
    """Token id -> UTF-8 bytes, plus the id that signals end-of-sequence."""

    token_bytes: tuple[bytes, ...]
    eos_id: int | None = None

    def __len__(self) -> int:
        return len(self.token_bytes)

    @property
    def size(self) -> int:
        return len(self.token_bytes)

    def decode(self, ids: list[int]) -> bytes:
        return b"".join(self.token_bytes[i] for i in ids)

    @classmethod
    def from_tokens(
        cls, tokens: list[str] | list[bytes], eos_id: int | None = None
    ) -> Vocabulary:
        tb = tuple(t if isinstance(t, bytes) else t.encode("utf-8") for t in tokens)
        return cls(token_bytes=tb, eos_id=eos_id)

    @classmethod
    def from_hf(cls, tokenizer_or_name: object = DEFAULT_MODEL) -> Vocabulary:
        try:
            from transformers import AutoTokenizer  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Vocabulary.from_hf needs `transformers`; install bpdecode[hf]"
            ) from exc

        tok = (
            tokenizer_or_name
            if hasattr(tokenizer_or_name, "convert_ids_to_tokens")
            else AutoTokenizer.from_pretrained(tokenizer_or_name)  # type: ignore[arg-type]
        )
        vocab_size = int(getattr(tok, "vocab_size", len(tok)))
        byte_decoder = _gpt2_byte_decoder(tok)
        added = set(getattr(tok, "get_added_vocab", dict)().values())

        out: list[bytes] = []
        for i in range(vocab_size):
            piece = tok.convert_ids_to_tokens(i)
            if piece is None:
                out.append(b"")
                continue
            if i in added or byte_decoder is None:
                out.append(tok.decode([i]).encode("utf-8"))
                continue
            try:
                out.append(bytes(byte_decoder[c] for c in piece))
            except KeyError:
                out.append(tok.decode([i]).encode("utf-8"))

        if byte_decoder is None:
            warnings.warn(
                "tokenizer is not GPT-2 byte-level; token->bytes mapping uses "
                "decode() and may be imprecise for whitespace/partial code points",
                stacklevel=2,
            )
        return cls(token_bytes=tuple(out), eos_id=getattr(tok, "eos_token_id", None))


def _gpt2_byte_decoder(tok: object) -> dict[str, int] | None:
    """Return the GPT-2 char<->byte table if this tokenizer uses it."""
    backend = getattr(tok, "backend_tokenizer", None)
    model = getattr(backend, "model", None)
    if model is None or type(model).__name__ != "BPE":
        return None
    # Reconstruct the standard GPT-2 byte<->unicode mapping.
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs, strict=True)}
