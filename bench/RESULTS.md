# regex_mask.py -- recorded run

`Qwen/Qwen2.5-0.5B` tokenizer (V = 151 643), **CPU** (Apple M-series),
`outlines_core` 0.2.14, median of 5 runs, batch 8.

| case | build bp | build bp +dense | build ol | step bp (byte-walk) | step bp (dense) | step ol | disagree |
|---|---:|---:|---:|---:|---:|---:|---:|
| email    | 16 ms | 39 ms | 10 ms |  589 µs | **112 µs** | 206 µs | 0 / 3 |
| ipv4     | 16 ms | 59 ms | 11 ms |  283 µs |   92 µs |   2.3 µs | 0 / 12 |
| sentence | 16 ms | 39 ms | 11 ms | 1223 µs | **137 µs** | 401 µs | 0 / 6 |
| json     | 17 ms | 74 ms | 17 ms |  414 µs |   94 µs |  32 µs | 0 / 13 |

`step` is per token **per sequence** (µs).

## Reading

**Correctness: identical.** Zero disagreements on the non-EOS allowed set across
every case -- bpdecode's byte-DFA path and Outlines' token FSM agree token for
token.

**Compile: bpdecode ~16 ms, dense adds 20-60 ms.** Both within a few x of
`outlines_core`. The dense table (`build_token_transitions`) runs the whole
vocab through the byte-DFA; walking tokens shortest-first keeps the cost near
`sum(lengths)` rather than `V x max_len`. `GrammarCache`d, so it is a
time-to-first-token cost only.

**Per-token mask:**

- **byte-walk path** (`step()` over the whole vocab each step): 3-15x slower
  than Outlines. Outlines' `Index` precomputes the token-level transition
  table, so its step is a memcpy of a cached bitmask row.
- **dense path** (`FsaTensors(dense=True)` -- our version of that same
  precomputed table, `tok_next[state][token]`): a step is `tok_next[states]
  != -1` then `masked_fill_`. **Faster than Outlines on the non-trivial
  patterns** (email, sentence), slower on the trivial ones where Outlines is
  just copying a ~5 KB row. `dense="auto"` (the default) turns it on when
  `num_states x vocab` fits in ~32 M entries.

This is still **CPU**, where a step is `[batch, 151 k]` tensor ops with no real
parallelism. Rerun `--device cuda --batch 64` on the GPU box for the comparison
that matters.

**The bet.** Hard-masking is a solved problem. bpdecode's dense path is now
competitive; the actual differentiator is the Phase 4 soft-lookahead layer
(a weighted backward pass Outlines/XGrammar don't have) and GPU batch
amortization.
