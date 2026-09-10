# regex_mask.py -- recorded run

`Qwen/Qwen2.5-0.5B` tokenizer (V = 151 643), **CPU** (Apple M-series),
`outlines_core` 0.2.14, median of 5 runs, batch 8.

| case | build bp | build ol | step bp (kernel) | step bp (cache) | step ol | disagree |
|---|---:|---:|---:|---:|---:|---:|
| email    | 17 ms | 10 ms |  584 µs | 593 µs | 207 µs | 0 / 3 |
| ipv4     | 17 ms | 11 ms |  276 µs | 303 µs |  2.3 µs | 0 / 12 |
| sentence | 17 ms | 12 ms | 1222 µs | 1228 µs | 412 µs | 0 / 6 |
| json     | 17 ms | 22 ms |  432 µs | 412 µs |  34 µs | 0 / 13 |

`step` is per token **per sequence** (µs). Earlier runs had `build bp` at
~200 ms; `token_symbols` now maps token bytes through a 256-entry
`bytes.translate` table instead of a Python loop over 151 k tokens.

## Reading

**Correctness: identical.** Zero disagreements on the non-EOS allowed set across
every case -- bpdecode's byte-DFA path and Outlines' token FSM agree token for
token.

**Compile: at parity.** Both ~10-22 ms; bpdecode wins on `json`, loses on the
simpler patterns. The regex -> DFA subset construction is 0.4 ms; the rest is
turning the tables into tensors. `GrammarCache`d anyway, so it is a
time-to-first-token cost only.

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

**The bet.** Hard-masking is a solved problem and mature FSM libs are fast at
it. bpdecode's differentiator is the Phase 4 soft-lookahead layer (a weighted
backward pass Outlines/XGrammar don't have) and GPU batch amortization -- not
out-teching `outlines_core` at bitmask memcpy on CPU.
