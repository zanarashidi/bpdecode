"""bpdecode vs Outlines: grammar compile time and per-token mask overhead.

    pip install outlines transformers        # plus bpdecode, torch
    python bench/regex_mask.py               # CPU; add --device cuda on a GPU box

Measures two things that matter for a serving loop:

1.  build   -- regex string -> ready-to-mask (one-time per distinct grammar)
2.  step    -- produce the allow-mask for one decode step, at batch size B

Both libraries expose "allowed token ids for the current automaton state"; EOS
handling and tokenizer edge cases differ slightly, so the step numbers are a
throughput comparison, not a claim of identical semantics.  A correctness
cross-check reports how often the allowed sets disagree.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import torch

CASES: list[tuple[str, str, str]] = [
    ("email", r"[a-z]+@[a-z]+\.[a-z]+", "user@example.com"),
    (
        "ipv4",
        r"[0-9][0-9]?[0-9]?\.[0-9][0-9]?[0-9]?\.[0-9][0-9]?[0-9]?\.[0-9][0-9]?[0-9]?",
        "192.168.1.10",
    ),
    ("sentence", r"[A-Z][a-z]+( [a-z]+)+\.", "The quick brown fox jumps."),
    (
        "json",
        r'\{"name": "[a-z]+", "id": [0-9]+\}',
        '{"name": "alice", "id": 42}',
    ),
]


def _timed(fn, repeat: int) -> float:
    """Median seconds over `repeat` runs."""
    xs = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        xs.append(time.perf_counter() - t)
    return statistics.median(xs)


@dataclass
class Row:
    case: str
    build_bp: float
    build_ol: float
    step_bp_kernel: float
    step_bp_cached: float
    step_ol: float
    disagree: int
    steps: int


def run(model: str, device: str, batch: int, repeat: int) -> list[Row]:
    import outlines_core as oc
    from transformers import AutoTokenizer

    from bpdecode.batch import ConstraintBatch
    from bpdecode.ops import FsaTensors
    from bpdecode.tokenizer import Vocabulary

    hf_tok = AutoTokenizer.from_pretrained(model)
    bp_vocab = Vocabulary.from_hf(hf_tok)
    ol_vocab = oc.Vocabulary.from_pretrained(model)
    V = bp_vocab.size
    words_ol = (len(ol_vocab) + 1 + 31) // 32  # outlines sizes for len+1 (EOS)

    def bench_case(name: str, pattern: str, sample: str) -> Row:
        build_bp = _timed(lambda: FsaTensors.build(pattern, bp_vocab), repeat)
        build_ol = _timed(lambda: oc.Index(pattern, ol_vocab), repeat)

        fsa = FsaTensors.build(pattern, bp_vocab).to(device)
        token_ids = hf_tok(sample, add_special_tokens=False).input_ids
        steps = len(token_ids)

        # correctness cross-check: walk the sample, batch 1
        disagree = 0
        cb1 = ConstraintBatch(fsa, capacity=1, device=device)
        cb1.add(0)
        guide = oc.Guide(oc.Index(pattern, ol_vocab))
        for tid in token_ids:
            row = torch.zeros(1, V, device=device)
            cb1.apply_mask([0], row)
            bp_allowed = set((row[0] == 0.0).nonzero().flatten().tolist())
            ol_allowed = set(guide.get_tokens())
            bp_allowed.discard(bp_vocab.eos_id)
            ol_allowed.discard(ol_vocab.get_eos_token_id())
            disagree += len(bp_allowed ^ ol_allowed) > 0
            cb1.commit([0], torch.tensor([tid], dtype=torch.int32, device=device))
            guide.advance(tid)

        def bp_step(cache: bool):
            cb = ConstraintBatch(fsa, capacity=batch, device=device, mask_cache=cache)
            ids = list(range(batch))
            for i in ids:
                cb.add(i)
            logits = torch.zeros(batch, V, device=device)
            toks = [
                torch.full((batch,), t, dtype=torch.int32, device=device)
                for t in token_ids
            ]

            def one() -> None:
                for t in toks:
                    cb.apply_mask(ids, logits)
                    cb.commit(ids, t)
                for i in ids:
                    cb.reset(i)

            return one

        def ol_step():
            guides = [oc.Guide(oc.Index(pattern, ol_vocab)) for _ in range(batch)]
            buf = torch.zeros(batch, words_ol, dtype=torch.int32)

            def one() -> None:
                for tid in token_ids:
                    for r, g in enumerate(guides):
                        g.write_mask_into(
                            buf[r].data_ptr(), buf[r].numel(), buf[r].element_size()
                        )
                        g.advance(tid)
                for g in guides:
                    g.reset()

            return one

        denom = steps * batch
        return Row(
            name,
            build_bp,
            build_ol,
            _timed(bp_step(False), repeat) / denom,
            _timed(bp_step(True), repeat) / denom,
            _timed(ol_step(), repeat) / denom,
            disagree,
            steps,
        )

    return [bench_case(*c) for c in CASES]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=7)
    args = ap.parse_args()

    rows = run(args.model, args.device, args.batch, args.repeat)

    print(f"\nmodel={args.model}  device={args.device}  batch={args.batch}\n")
    print(
        f"{'case':<10} {'build bp':>10} {'build ol':>10}   "
        f"{'step bp/k':>10} {'step bp/c':>10} {'step ol':>10}   {'disagree':>9}"
    )
    print("-" * 88)
    for r in rows:
        print(
            f"{r.case:<10} {r.build_bp * 1e3:>9.1f}m {r.build_ol * 1e3:>9.1f}m   "
            f"{r.step_bp_kernel * 1e6:>9.1f}u {r.step_bp_cached * 1e6:>9.1f}u "
            f"{r.step_ol * 1e6:>9.1f}u   {r.disagree:>4}/{r.steps:<4}"
        )
    print(
        "\nbuild = regex -> ready (ms).  step = per token per sequence "
        "(us).  bp/k = kernel, bp/c = mask cache.\n"
        "disagree = decode steps where the non-EOS allowed sets differ."
    )


if __name__ == "__main__":
    main()
