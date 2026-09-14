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

from bpdecode.lookahead import generate_model_weighted, generate_model_weighted_cfg
from bpdecode.tokenizer import Vocabulary

ENTITIES = [
    "a jazz musician from New Orleans",
    "a mountain in the Alps",
    "a programming language",
    "a species of owl",
    "a chess opening",
    "a dessert from France",
]
PATTERN = r'\{"name": "[A-Za-z][A-Za-z .]+", "year": [0-9]+\}'  # unbounded name, regular

# same shape, but expressed as a recursive CFG rule (list of tags) so it
# exercises the on-device PDA path (grammar.device) instead of FsaTensors --
# a regex can't express "zero or more comma-separated tags". No free-standing
# `ws ::= " "*` production: a small greedy model has no reason to ever stop
# padding an unbounded whitespace loop, hard mask or not (that failure mode
# is a different thing from the token-level dead ends this module targets --
# see test_grammar_hf.py's identical caveat about flat-bias greedy decoding).
CFG_GRAMMAR = (
    'root ::= "{\\"name\\": \\"" [A-Za-z][A-Za-z ]* "\\", \\"tags\\": [" tags "]}"\n'
    'tags ::= "" | tag ("," tag)*\n'
    'tag ::= "\\"" [a-z]+ "\\""'
)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--grammar-mode", choices=["regex", "cfg"], default="regex")
    ap.add_argument(
        "--repeat", type=int, default=1,
        help="duplicate the entity list this many times, to test whether a "
        "bigger batch amortizes per-step launch overhead better on GPU",
    )
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
        .eval()
        .to(args.device)
    )
    vocab = Vocabulary.from_hf(tok)
    entities = ENTITIES * args.repeat
    prompts = [f"Give one JSON object describing {e}. JSON: " for e in entities]

    if args.grammar_mode == "regex":
        pattern = PATTERN
        gen = generate_model_weighted
    else:
        pattern = CFG_GRAMMAR
        gen = generate_model_weighted_cfg

    configs = [
        (0.0, "hard mask + dead-end check (alpha=0)"),
        (args.alpha, f"model-weighted (alpha={args.alpha})"),
    ]
    for alpha, label in configs:
        t0 = time.time()
        out = gen(
            model, tok, prompts, pattern, vocab=vocab, k=args.k, alpha=alpha,
            max_new_tokens=args.max_new_tokens, device=args.device,
        )
        dt = time.time() - t0
        if args.grammar_mode == "regex":
            complete = sum(bool(re.match(PATTERN, t)) for t in out)
        else:
            complete = len(out)
        steps = f"{len(prompts)} rows x {args.max_new_tokens} steps"
        per_step_ms = dt / args.max_new_tokens * 1000
        header = f"{label} [{args.grammar_mode}]  ({dt:.1f}s for {steps}, {per_step_ms:.1f}ms/step)"
        print(f"\n=== {header} ===")
        print(f"complete: {complete}/{len(out)}")
        for t in out[:6]:
            ok = bool(re.match(PATTERN, t)) if args.grammar_mode == "regex" else True
            print(f"  {'OK ' if ok else 'INC'}  {t!r}")
        if len(out) > 6:
            print(f"  ... ({len(out) - 6} more rows omitted)")


if __name__ == "__main__":
    main()
