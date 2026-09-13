"""Batched, KV-cache-reusing model-weighted lookahead
(bpdecode.lookahead.generate_model_weighted) against plain hard masking.

Generalizes bench/soft_eval_modelweighted.py's single-request, no-cache
proof of concept into the real batched generation loop -- see
bpdecode/lookahead.py's docstring for how the cache forking works and
bench/RESULTS.md for the numbers this produces.

    pip install transformers torch        # plus bpdecode
    python bench/lookahead_model_weighted.py
"""

from __future__ import annotations

import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bpdecode.lookahead import generate_model_weighted
from bpdecode.tokenizer import Vocabulary

ENTITIES = [
    "a jazz musician from New Orleans",
    "a mountain in the Alps",
    "a programming language",
    "a species of owl",
    "a chess opening",
    "a dessert from France",
]
PATTERN = r'\{"name": "[A-Za-z][A-Za-z .]+", "year": [0-9]+\}'  # unbounded name


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
        .eval()
        .to(args.device)
    )
    vocab = Vocabulary.from_hf(tok)
    prompts = [f"Give one JSON object describing {e}. JSON: " for e in ENTITIES]

    configs = [
        (0.0, "hard mask + dead-end check (alpha=0)"),
        (args.alpha, f"model-weighted (alpha={args.alpha})"),
    ]
    for alpha, label in configs:
        t0 = time.time()
        out = generate_model_weighted(
            model, tok, prompts, PATTERN, vocab=vocab, k=args.k, alpha=alpha,
            max_new_tokens=args.max_new_tokens, device=args.device,
        )
        dt = time.time() - t0
        complete = sum(bool(re.match(PATTERN, t)) for t in out)
        steps = f"{len(prompts)} rows x {args.max_new_tokens} steps"
        print(f"\n=== {label}  ({dt:.1f}s for {steps}) ===")
        print(f"complete: {complete}/{len(out)}")
        for t in out:
            print(f"  {'OK ' if re.match(PATTERN, t) else 'INC'}  {t!r}")


if __name__ == "__main__":
    main()
