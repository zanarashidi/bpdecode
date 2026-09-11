# Summary

All CPU, `Qwen/Qwen2.5-0.5B` tokenizer (~151k tokens). Numbers are per token
per sequence.

| | compile | per-token mask (steady state) | correctness |
|---|---|---|---|
| **regex** vs `outlines_core` | ~17 ms (parity) | dense **~90-140 µs**, beats Outlines on non-trivial patterns | identical allowed sets |
| **JSON Schema** vs `xgrammar` / `llguidance` | ~1 ms | memoised **~0.6 µs** -- ~14x under xgrammar, ~70x under llguidance | valid, tested vs `jsonschema` |
| **soft lookahead** | -- | -- | **negative result**: count-based weighting over-extends, doesn't beat hard masking |

The steady-state numbers assume a warm cache: the first request with a new
grammar pays ~0.2 s, and a token trie is built once per vocabulary (~0.5 s).
Both are one-time and shared across all requests.

## GPU (RTX 3090, batch 64)

`gpu_check.sh` -- kernels validated (8 gtests, 22-case CUDA-vs-CPU
differential, `compute-sanitizer` clean) and:

| | per-token mask |
|---|---|
| regex, dense `tok_next` gather | **~8 µs, flat across patterns** (email 9.9, ipv4 7.9, sentence 8.5, json 7.7); Outlines: 312 / 4.3 / 560 / 54 |
| regex, byte-walk CUDA kernel (non-dense fallback) | 61-134 µs |
| JSON Schema, warm | **~1 µs**; xgrammar 6-14 µs, llguidance 53-64 µs |

On GPU the dense gather is a fixed ~8 µs regardless of grammar complexity --
it beats `outlines_core` on every non-trivial pattern and stays ~2 µs behind
only on `ipv4` (a 5-state FSM where Outlines is a bare memcpy).

Detail follows.

---

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

---

# json_schema_mask.py -- bpdecode vs XGrammar vs llguidance

Same box, `xgrammar` 0.2.6, `llguidance` 1.8.0. Per token per sequence unless
noted. bpdecode also builds a token trie once per vocabulary (~0.5 s).

| schema | metric | bpdecode | xgrammar | llguidance |
|---|---|---:|---:|---:|
| person | compile | 0.5 ms | ~0 ms | 0.1 ms |
| person | **cold** (1st request) | ~9000 µs | 14 µs | 121 µs |
| person | **warm** (steady state) | **0.6 µs** | 8.3 µs | 43 µs |
| nested | compile | 0.6 ms | ~0 ms | 0.1 ms |
| nested | cold | ~6800 µs | 5 µs | 62 µs |
| nested | warm | **0.6 µs** | 3.8 µs | 56 µs |

## Reading

bpdecode's config-set masks (and the residual DFAs for regular loops like
`json-char*`) are **memoised on the compiled grammar**, keyed by config-set and
shared across every request. So:

- **first request** with a new grammar pays a warmup -- ~0.15-0.2 s total
  (the "cold" per-token figure is that spread over the sample). Plus the
  one-time ~0.5 s token-trie build per vocabulary.
- **every request after** is a dict lookup: **~0.6 µs / token**, an order of
  magnitude under xgrammar and ~70x under llguidance's Python API.

xgrammar does the equivalent precompute in C++ at matcher-creation time, so its
cold path is already fast; it has no warm speedup to give. For a serving
workload where one schema handles thousands of requests, bpdecode's warm number
is what matters. The warmup is the price; hiding it (precompute at compile,
move the trie build to the C++ core) is Phase 5.

---

# soft_eval.py -- soft lookahead vs hard masking (Phase 4)

Qwen2.5-0.5B, greedy, 10 "describe X as JSON" prompts, constrained to a regex
for `{"name": "...", "year": N}`. `k = 3` backward steps.

| pattern | config | complete | mean name len | repeated-word rate |
|---|---|---:|---:|---:|
| name `{1,30}` | hard | 1.00 | 11.6 | 0.10 |
| name `{1,30}` | soft α=+0.5 | 1.00 | 28.7 | 0.30 |
| name `{1,30}` | soft α=+1.0 | 1.00 | 28.9 | 0.40 |
| name `{1,30}` | soft α=-0.3 | 1.00 | 7.4 | 0.00 |
| name `{1,30}` | soft α=-1.0 | 1.00 | 17.3 | 0.00 |
| name `[A-Za-z .]+` (unbounded) | hard | 1.00 | 11.6 | 0.10 |
| name `[A-Za-z .]+` | soft α=+0.5 | **0.00** | -- | -- |
| name `[A-Za-z .]+` | soft α=+1.0 | **0.00** | -- | -- |
| name `[A-Za-z .]+` | soft α=-0.3 | 1.00 | 7.4 | 0.00 |
| name `[A-Za-z .]+` | soft α=-1.0 | 1.00 | 4.5 | 0.00 |

Sample names (unbounded): hard -> `Louis Armstrong`, `Mont Blanc`, `Python`;
α=+1.0 -> `Mont Blanc  romeo  romeo  romeo`; α=-1.0 -> `Louis`, `Mount`, `Python`.
(α=-1.0 on the *bounded* pattern occasionally lets a single long degenerate
token through -- `NavigationItemSelectedListener` -- since a strong terseness
bias with only one long completing token still admits it.)

## Reading -- the hypothesis does not hold

`build_lookahead` counts grammar-valid token continuations. The plan's idea was
to bias **toward** states with more continuations (α > 0), steering away from
valid-but-dead-end tokens.

- **α > 0 makes it worse.** "More continuations" is maximised by *not
  finishing*: a field that can accept 20 more characters outscores one that can
  stop now. So positive α pads bounded fields to their limit with repetitive
  filler (`romeo romeo romeo`) and **never terminates unbounded ones**
  (0% completion).
- **α < 0 (bias toward completion)** keeps every output valid and makes them
  more concise, but truncates real content (`Louis` for `Louis Armstrong`).
  It is a usable "terseness" knob, not an accuracy win -- hard masking plus a
  competent model already produces the full name.
- The token-level formulation *does* correctly prune tokens with no valid
  *token* continuation (the Phase 3 partial-token dead end) -- that part works
  and is kept.

**Why:** uniform continuation-counting ignores the model's own distribution.
The continuations soft lookahead rewards are mostly ones the model would never
generate. A useful lookahead has to weight continuations by model probability
(proper model-predictive control / lookahead decoding), which needs extra
forward passes -- a heavier technique, out of scope here.

**Verdict:** the count-based soft-lookahead bias is not a win over hard masking
for structured output. The machinery ships (it is the same backward
sum-product as `build_reachability`, and the API / knob are there for
model-weighted experiments), with this negative result recorded.

---

# soft_eval_modelweighted.py -- follow-up: is model probability the fix?

Same setup (Qwen2.5-0.5B, greedy, unbounded `[A-Za-z .]+` name field, where
count-based soft lookahead got 0/6 complete). Replaces the uniform
continuation count with a **1-step model-weighted** lookahead: take the top-K
(K=6) grammar-allowed candidates by the current logit, run one batched extra
forward pass to get each candidate's next-step distribution, and score by how
much of *that* distribution lands on a grammar-valid continuation --
`score(t) = logit(t) + alpha * log P(next token grammar-valid | took t)`. Cost
is K extra forward passes per step, independent of how many tokens the
grammar allows (unlike the k-step automaton sum-product).

| config | complete | sample |
|---|---:|---|
| hard masking | 6/6 | `Louis Armstrong`, `Mont Blanc`, `Python`, `Bubo bubo` |
| count-based soft (α=+0.5) | 0/6 | (never terminates) |
| **model-weighted 1-step (α=1.0)** | **6/6** | `Louis Armstrong`, `Mont Blanc`, `Python`, `Bubo bubo`, `Sicilian Defense`, `Chiffon` |

Identical to hard masking on every case here (the model already wanted these
completions), and -- unlike the count-based version -- **no padding
degeneration** on the bounded-length pattern either.

**Reading:** the Phase 4 hypothesis was right about the *shape* of the fix
(bias the mask toward better continuations) but wrong about the *signal*
(raw continuation count vs. the model's own probability). Weighting by real
next-token probability, even just one step ahead, removes the over-extension
pathology while keeping the token-level structural pruning (a candidate whose
only continuations are grammar-dead still scores `-inf`, via the same masking
machinery). The cost is real -- K extra forward passes per decode step -- so
this is a mode for cases that need the guidance (weak models, tricky
schemas), not a default. A full k-step version is the natural next step if
this is worth productionising; not built here.
