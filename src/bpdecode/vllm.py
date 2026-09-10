"""vLLM logits-processor adapter.

:class:`RegexLogitsProcessor` is a **request-level** processor -- one instance
per request, passed via ``SamplingParams(logits_processors=[proc])``.  This is
a supported vLLM feature on V0 and on V1 (with request-level processors
enabled).  Call signature is vLLM's 2-arg form ``(past_token_ids, logits)``:
``past_token_ids`` is the running list of *generated* ids, ``logits`` a 1-D
``[vocab]`` tensor.

    from bpdecode.vllm import RegexLogitsProcessorFactory
    from vllm import LLM, SamplingParams

    llm = LLM("Qwen/Qwen2.5-0.5B")
    factory = RegexLogitsProcessorFactory(llm.get_tokenizer())
    params = SamplingParams(
        logits_processors=[factory.make(r"\\d{4}-\\d\\d-\\d\\d")],
        max_tokens=32,
    )
    llm.generate("The date is ", params)

For the V1 batch-level processor
(``vllm.v1.sample.logits_processor.LogitsProcessor``), drive a
:class:`~bpdecode.batch.ConstraintBatch` from ``update_state`` /  ``apply``:
:class:`BatchConstraintState` is that state machine, minus vLLM's version-
specific ``BatchUpdate`` shape.  This module never imports ``vllm``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .batch import ConstraintBatch, GrammarCache
from .ops import FsaTensors
from .regex.compile import DFA
from .tokenizer import Vocabulary


def _as_vocabulary(tokenizer_or_vocab: object) -> Vocabulary:
    if isinstance(tokenizer_or_vocab, Vocabulary):
        return tokenizer_or_vocab
    return Vocabulary.from_hf(tokenizer_or_vocab)


class RegexLogitsProcessor:
    """One per request; constrains the generated continuation to ``pattern``."""

    def __init__(
        self,
        pattern: str | DFA,
        tokenizer: object,
        *,
        device: torch.device | str = "cpu",
        neg_inf: float = float("-inf"),
        _fsa: FsaTensors | None = None,
    ) -> None:
        fsa = _fsa or FsaTensors.build(pattern, _as_vocabulary(tokenizer))
        self._batch = ConstraintBatch(fsa, capacity=1, device=device)
        self._batch.add(0)
        self._neg_inf = neg_inf
        self._pos = 0

    def __call__(
        self, past_token_ids: Sequence[int], logits: torch.Tensor
    ) -> torch.Tensor:
        n = len(past_token_ids)
        if n < self._pos:  # sequence rewound (shouldn't happen) -- replay
            self._batch.reset(0)
            self._pos = 0
        while self._pos < n:
            tok = int(past_token_ids[self._pos])
            self._batch.commit([0], torch.tensor([tok], dtype=torch.int32))
            self._pos += 1

        row = logits.unsqueeze(0)
        self._batch.apply_mask([0], row, self._neg_inf)
        return row.squeeze(0)

    @property
    def is_complete(self) -> bool:
        return self._batch.is_complete(0)


class RegexLogitsProcessorFactory:
    """Compile each distinct pattern once; hand out fresh per-request processors."""

    def __init__(
        self, tokenizer: object, *, device: torch.device | str = "cpu"
    ) -> None:
        self._cache = GrammarCache(_as_vocabulary(tokenizer), device=device)
        self._device = device

    def make(self, pattern: str) -> RegexLogitsProcessor:
        return RegexLogitsProcessor(
            pattern, None, device=self._device, _fsa=self._cache.get(pattern)
        )


class BatchConstraintState:
    """The state machine behind a vLLM V1 batch-level ``LogitsProcessor``.

    vLLM's persistent batch addresses each running request by a slot index that
    is reused as requests finish and is occasionally reshuffled.  Feed those
    events in (``add`` / ``remove`` / ``move``) and the freshly sampled tokens
    (``advance``); call :meth:`mask` on the ``[num_rows, vocab]`` logits.

    One grammar per batch (the common serving case -- every request against the
    same schema); a second pattern raises.
    """

    def __init__(
        self,
        tokenizer: object,
        *,
        device: torch.device | str = "cpu",
        mask_cache: bool = True,
        capacity: int = 256,
    ) -> None:
        self._grammars = GrammarCache(_as_vocabulary(tokenizer), device=device)
        self._device = device
        self._mask_cache = mask_cache
        self._capacity = capacity
        self._batch: ConstraintBatch | None = None
        self._pattern: str | None = None
        self._pos: dict[int, int] = {}

    def add(self, row: int, pattern: str) -> None:
        if self._batch is None:
            self._pattern = pattern
            self._batch = ConstraintBatch(
                self._grammars.get(pattern),
                capacity=self._capacity,
                device=self._device,
                mask_cache=self._mask_cache,
            )
        elif pattern != self._pattern:
            raise NotImplementedError(
                f"one grammar per batch; got {pattern!r} vs {self._pattern!r}"
            )
        if row in self._pos:
            self._batch.reset(row)
        else:
            self._batch.add(row)
        self._pos[row] = 0

    def remove(self, row: int) -> None:
        if row in self._pos:
            self._batch.evict(row)  # type: ignore[union-attr]
            del self._pos[row]

    def move(self, src: int, dst: int) -> None:
        if src not in self._pos:
            return
        assert self._batch is not None
        state = self._batch.state_of(src)
        self._batch.evict(src)
        self._batch.add(dst)
        self._batch._state[self._batch._slot_of[dst]] = state
        self._pos[dst] = self._pos.pop(src)

    def advance(self, row: int, generated_ids: Sequence[int]) -> None:
        """``generated_ids`` is the request's full output-token list so far."""
        if row not in self._pos or self._batch is None:
            return
        for tok in generated_ids[self._pos[row] :]:
            self._batch.commit([row], torch.tensor([int(tok)], dtype=torch.int32))
        self._pos[row] = len(generated_ids)

    def mask(self, logits: torch.Tensor, neg_inf: float = float("-inf")) -> torch.Tensor:
        """Mask the rows this state manages, in place. ``logits`` is
        ``[num_rows, vocab]`` indexed by slot; rows not managed are untouched.
        """
        if not self._pos or self._batch is None:
            return logits
        rows = sorted(self._pos)
        idx = torch.tensor(rows, device=logits.device)
        sub = logits.index_select(0, idx)
        self._batch.apply_mask(rows, sub, neg_inf)
        logits[idx] = sub
        return logits

    def is_complete(self, row: int) -> bool:
        return self._batch is not None and self._batch.is_complete(row)
