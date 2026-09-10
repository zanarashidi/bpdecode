# regex_mask.py -- recorded run

`Qwen/Qwen2.5-0.5B` tokenizer (V = 151 643), **CPU** (Apple M-series),
`outlines_core` 0.2.14, median of 4-5 runs.

| case | build bp | build ol | step bp (kernel) | step bp (cache) | step ol | disagree |
|---|---:|---:|---:|---:|---:|---:|
| **batch 1** |
| email    | 195 ms | 10 ms | 1065 µs | 964 µs | 216 µs | 0 / 3 |
| ipv4     | 172 ms | 11 ms |  739 µs | 731 µs |  2.6 µs | 0 / 12 |
| sentence | 194 ms | 11 ms | 1669 µs | 1647 µs | 397 µs | 0 / 6 |
| json     | 266 ms | 17 ms |  890 µs | 816 µs |  33 µs | 0 / 13 |
| **batch 8** |
| email    | 201 ms | 11 ms |  626 µs | 599 µs | 210 µs | 0 / 3 |
| ipv4     | 177 ms | 11 ms |  297 µs | 306 µs |  2.5 µs | 0 / 12 |
| sentence | 198 ms | 12 ms | 1248 µs | 1281 µs | 433 µs | 0 / 6 |
| json     | 278 ms | 18 ms |  416 µs | 420 µs |  33 µs | 0 / 13 |
| **batch 32** |
| email    | 195 ms | 11 ms |  541 µs | 569 µs | 203 µs | 0 / 3 |
| ipv4     | 174 ms | 11 ms |  230 µs | 224 µs |  2.5 µs | 0 / 12 |
| sentence | 194 ms | 11 ms | 1177 µs | 1209 µs | 408 µs | 0 / 6 |
| json     | 270 ms | 18 ms |  365 µs | 364 µs |  33 µs | 0 / 13 |

`step` is per token **per sequence** (µs).

## Reading

**Correctness: identical.** Zero disagreements on the non-EOS allowed set across
every case and batch size -- bpdecode's byte-DFA path and Outlines' token FSM
agree token for token.

**Compile: Outlines ~15-20x faster.** `outlines_core` is Rust; bpdecode's
compile (`compile_regex` subset construction + `token_symbols` over 151 k tokens)
is pure Python. This is one-time per distinct grammar and `GrammarCache`d, but
it hurts time-to-first-token. Fix: move the regex -> `FsaTable` +
`TokenSymbols` build into the C++ core (Phase 5), or a Rust/pybind path.

**Per-token mask: Outlines 2-15x faster on CPU.** Root cause is architectural:
Outlines' `Index` precomputes the *entire* token-level transition table up
front, so a decode step is a memcpy of a cached bitmask row. bpdecode recomputes
`step()` across the whole vocab every step (walking each token's bytes through
the byte-DFA). `MaskCache` barely helps here because a single sequence's walk
visits mostly-distinct states.

Two things to note:

1. This microbench runs on **CPU**, where bpdecode's "kernel" is a scalar loop
   with no parallelism -- the worst case for a design built around one GPU
   launch masking a whole batch. Rerun with `--device cuda --batch 64` on the
   GPU box for the comparison that matters.
2. The precomputed token-transition table is the "byte-DFA x tokenizer product"
   from the plan; Phase 1 shipped the *lazy* version. A dense
   `tok_next[state][token]` (≈ states x 151 k x 4 B, a few MB for a regex) would
   make a step an O(vocab) gather instead of O(vocab x bytes/token x log
   symbols). Candidate for Phase 2 follow-up or Phase 5.

**The bet.** Hard-masking is a solved problem and mature Rust FSM libs are
fast at it. bpdecode's differentiator is the Phase 4 soft-lookahead layer
(a weighted backward pass Outlines/XGrammar don't have) and GPU batch
amortization -- not out-teching `outlines_core` at bitmask memcpy.
