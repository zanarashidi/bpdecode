"""Model-weighted 1-step lookahead: does weighting by the model's own
next-token probability (instead of uniform continuation counts) fix the
negative result from `bpdecode.ops.build_lookahead` / `apply_soft_` (see
bench/RESULTS.md)?

Bounded experiment, not core library code: at each step, take the top-K
grammar-allowed candidates by the current logit, run ONE extra batched forward
pass to see each candidate's next-step distribution, and score each candidate
by how much of *that* distribution lands on a grammar-valid continuation:

    score(t) = logit(t) + alpha * log P(next token is grammar-valid | took t)

This costs K extra forward passes per decode step (batched into one call),
regardless of how many tokens the grammar allows -- unlike the k-step
automaton sum-product in `build_lookahead`, the weight comes from the model,
not a count.

    pip install transformers torch        # plus bpdecode
    python bench/soft_eval_modelweighted.py
"""

from __future__ import annotations

import re

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from bpdecode.batch import ConstraintBatch, GrammarCache

ENTITIES = [
    "a jazz musician from New Orleans",
    "a mountain in the Alps",
    "a programming language",
    "a species of owl",
    "a chess opening",
    "a dessert from France",
]
PATTERN = r'\{"name": "[A-Za-z][A-Za-z .]+", "year": [0-9]+\}'  # unbounded name


def model_weighted_generate(
    model, tok, cache: GrammarCache, prompt: str, *, k: int, alpha: float,
    max_new_tokens: int = 40,
) -> str:
    ids = tok(prompt, return_tensors="pt").input_ids
    fsa = cache.get(PATTERN)
    batch = ConstraintBatch(fsa, capacity=1)
    batch.add(0)
    v = fsa.vocab_size

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(ids).logits[0, -1, :v].clone()
        batch.apply_mask([0], logits.unsqueeze(0))
        if torch.isneginf(logits).all():
            break

        topk = torch.topk(logits, k=min(k, int((~torch.isneginf(logits)).sum())))
        cand_ids = topk.indices  # [k']

        # states after each candidate, batched
        cb2 = ConstraintBatch(fsa, capacity=len(cand_ids))
        for r in range(len(cand_ids)):
            cb2.add(r)
        cb2.commit(list(range(len(cand_ids))), cand_ids.to(torch.int32))

        # one batched forward pass for all candidates' next step
        cand_seqs = torch.cat(
            [ids.expand(len(cand_ids), -1), cand_ids.unsqueeze(1)], dim=1
        )
        with torch.no_grad():
            next_logits = model(cand_seqs).logits[:, -1, :v].clone()
        cb2.apply_mask(list(range(len(cand_ids))), next_logits)  # mask the futures
        logprobs = F.log_softmax(next_logits, dim=-1)
        weight = torch.where(
            torch.isneginf(next_logits),
            torch.full_like(logprobs, float("-inf")),
            logprobs,
        ).logsumexp(dim=-1)  # [k']  log P(next is grammar-valid | took candidate)

        scored = logits[cand_ids] + alpha * weight
        best = cand_ids[int(scored.argmax())]

        if int(best) == fsa.eos_id:
            break
        batch.commit([0], torch.tensor([int(best)], dtype=torch.int32))
        ids = torch.cat([ids, torch.tensor([[int(best)]])], dim=1)

    prompt_len = tok(prompt, return_tensors="pt").input_ids.shape[1]
    return tok.decode(ids[0, prompt_len:], skip_special_tokens=True)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()

    from bpdecode.tokenizer import Vocabulary

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    cache = GrammarCache(Vocabulary.from_hf(tok))

    print(f"model={args.model}  k={args.k}  alpha={args.alpha}  pattern={PATTERN}\n")
    complete = 0
    for ent in ENTITIES:
        prompt = f"Give one JSON object describing {ent}. JSON: "
        text = model_weighted_generate(model, tok, cache, prompt, k=args.k, alpha=args.alpha)
        ok = bool(re.match(PATTERN, text))
        complete += ok
        print(f"  {'OK ' if ok else 'INC'}  {text!r}")
    print(f"\ncomplete: {complete}/{len(ENTITIES)}")


if __name__ == "__main__":
    main()
