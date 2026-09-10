"""A minimal continuous-batching serving loop -- no real model, just the
constraint engine.

    python examples/serve_batch.py

Shows how :class:`~bpdecode.batch.GrammarCache` and
:class:`~bpdecode.batch.ConstraintBatch` drive many sequences at once: requests
join and leave every step, and the whole batch is masked / advanced in one call.
Here a fake "model" samples uniformly from the still-allowed tokens.
"""

from __future__ import annotations

import random

import torch

from bpdecode.batch import ConstraintBatch, GrammarCache
from bpdecode.tokenizer import Vocabulary

TOKENS = ["{", "}", '"', "id", ":", ",", " ", "0", "1", "2", "3", "true", "false", "<eos>"]
VOCAB = Vocabulary.from_tokens(TOKENS, eos_id=len(TOKENS) - 1)
PATTERN = r'\{"id":[0-9]+\}'


def main() -> None:
    random.seed(0)
    cache = GrammarCache(VOCAB)
    batch = ConstraintBatch(cache.get(PATTERN), capacity=8)

    pending = [f"req-{i}" for i in range(12)]
    active: dict[str, list[str]] = {}
    step = 0

    while pending or active:
        step += 1
        # admit new requests up to capacity
        while pending and len(batch) < batch.capacity:
            rid = pending.pop(0)
            batch.add(rid)
            active[rid] = []

        ids = list(active)
        logits = torch.zeros(len(ids), VOCAB.size)
        batch.apply_mask(ids, logits)  # one call masks the whole batch

        # fake model: pick a random still-allowed token per row
        chosen = []
        for row in range(len(ids)):
            allowed = (logits[row] == 0.0).nonzero().flatten().tolist()
            chosen.append(random.choice(allowed))
        batch.commit(ids, torch.tensor(chosen, dtype=torch.int32))

        for rid, tid in zip(ids, chosen, strict=True):
            active[rid].append(TOKENS[tid])
            if tid == VOCAB.eos_id or batch.is_complete(rid) and random.random() < 0.4:
                text = "".join(t for t in active.pop(rid) if t != "<eos>")
                batch.evict(rid)
                print(f"step {step:2d}  {rid} done -> {text}")

    print(f"\nall {step} steps, batch never exceeded {batch.capacity} slots")


if __name__ == "__main__":
    main()
