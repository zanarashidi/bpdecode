"""Does soft lookahead beat hard masking? An honest, small eval.

Qwen2.5-0.5B generates a small JSON object (constrained by a regex) for a set
of entities, greedy-decoded, under hard masking and under k-step soft lookahead
at a few `alpha`. We report:

* complete   -- fraction that reached a full match before the token budget
* nonempty   -- fraction whose string field is not ""
* strlen     -- mean length of the string field
* parses     -- fraction that json.loads cleanly

Hypothesis: hard masking lets the model take the locally-easy exits (an empty
string, an early close) that a soft bias toward mass-rich continuations avoids.

Result (see bench/RESULTS.md): the hypothesis does **not** hold. Positive alpha
(toward mass-rich futures) over-extends -- it pads bounded fields to their max
with repetitive filler and never terminates unbounded ones. Negative alpha
(toward completion) keeps outputs valid and more concise but truncates real
content. Uniform continuation-counting is the wrong signal; model-probability-
weighted lookahead would be the real fix, at the cost of extra forward passes.

    pip install transformers torch        # plus bpdecode
    python bench/soft_eval.py
"""

from __future__ import annotations

import json
import re

ENTITIES = [
    "a jazz musician from New Orleans",
    "a mountain in the Alps",
    "a programming language",
    "a species of owl",
    "a Renaissance painter",
    "a board game",
    "a type of cloud",
    "a river in South America",
    "a chess opening",
    "a dessert from France",
]

# two shapes: a length-bounded name field, and an unbounded one
PATTERNS = {
    "bounded": r'\{"name": "[A-Za-z][A-Za-z ]{1,30}", "year": [0-9]{1,4}\}',
    "unbounded": r'\{"name": "[A-Za-z][A-Za-z .]+", "year": [0-9]+\}',
}


def _run(model_id: str) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from bpdecode.hf import RegexLogitsProcessor

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()

    configs = [
        ("hard", dict()),
        ("soft a=+0.5", dict(soft_k=3, alpha=0.5)),  # toward mass-rich futures
        ("soft a=+1.0", dict(soft_k=3, alpha=1.0)),
        ("soft a=-0.3", dict(soft_k=3, alpha=-0.3)),  # toward completion
        ("soft a=-1.0", dict(soft_k=3, alpha=-1.0)),
    ]

    for pname, pattern in PATTERNS.items():
        print(f"\n=== pattern: {pname}  {pattern}\n")
        print(f"{'config':<14}{'complete':>10}{'strlen':>9}{'repeat':>8}   sample")
        print("-" * 78)
        for label, kw in configs:
            complete = 0
            strlens, repeats, samples = [], [], []
            for ent in ENTITIES:
                prompt = f"Give one JSON object describing {ent}. JSON: "
                ids = tok(prompt, return_tensors="pt").input_ids
                lp = RegexLogitsProcessor(pattern, tok, **kw)
                with torch.no_grad():
                    out = model.generate(
                        ids,
                        logits_processor=[lp],
                        max_new_tokens=48,
                        do_sample=False,
                        pad_token_id=tok.eos_token_id,
                    )
                text = tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True)
                m = re.match(pattern, text)
                if m:
                    complete += 1
                    name = json.loads(m.group(0))["name"].strip()
                    strlens.append(len(name))
                    words = name.lower().split()
                    repeats.append(len(words) != len(set(words)))
                    if len(samples) < 3:
                        samples.append(name)
            n = len(ENTITIES)
            sl = sum(strlens) / len(strlens) if strlens else 0.0
            rp = sum(repeats) / len(repeats) if repeats else 0.0
            print(
                f"{label:<14}{complete / n:>10.2f}{sl:>9.1f}{rp:>8.2f}   "
                + " | ".join(samples)
            )


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    _run(ap.parse_args().model)


if __name__ == "__main__":
    main()
