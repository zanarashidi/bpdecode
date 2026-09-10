# bpdecode roadmap

GPU-accelerated constrained decoding: at every decode step, intersect the
model's vocab distribution with "which tokens keep generation on a
grammar-valid path", batched across a serving workload, without stalling the
GPU running the forward pass.

## Why this shape

| loopy-belief-propagation-using-CUDA | constrained decoding |
|---|---|
| factor-graph nodes | automaton states |
| `max_neigh` neighbour arrays | transition table (-> CSR) |
| boolean AND/NAND message passing | reachability: "can this state still reach accept" |
| sum-product `svf` messages | soft lookahead: mass of valid continuations |
| iterate-to-fixpoint + `atomicAnd` | backward reachability solve (once per grammar) |
| brute-force marginal reference | CPU reference parser (test oracle) |

Two layers:

1. **Hard mask** (boolean semiring) -- parity with Outlines / XGrammar.
   `mask[v] = 1` iff `delta(state, v)` is defined and lands in a live state.
2. **Soft lookahead** (sum-product) -- k-step weighted backward pass giving
   `log Z(valid continuations | state, v)` as a logit *bias*, steering the
   model away from valid-but-dead-end tokens. This is the novel bit.

## Module layout

```
src/bpdecode/     host front-end (Python): regex/PDA compilers, TokenDFA,
                  Constraint interface, CPU reference (correctness oracle)
  regex/utf8.py               code-point range -> UTF-8 byte automaton
  fsa.py                      DFA + Vocabulary -> FsaTable/TokenSymbols export
                              (+ scalar mirrors of the C++ core)
  ops.py                      torch.ops.bpdecode.* wrappers (FsaTensors bundle)
  batch.py                     GrammarCache / ConstraintBatch / MaskCache
  hf.py                        transformers RegexLogitsProcessor
  vllm.py                      vLLM request- / batch-level logits processors
  grammar/                     GBNF -> IR -> rule NFAs -> config-set PDA;
                               CFGConstraint (CPU oracle for CFGs)
csrc/             C++/CUDA core: FsaTable/TokenSymbols ABI, mask kernels
  include/bpdecode/mask.hpp    the ABI callers compile against
  src/mask_cpu.cpp             scalar reference + build_reachability
  src/mask_cuda.cu             batched kernels (warp/request, __ballot_sync)
bindings/torch_ops.cpp       ATen glue -> torch.ops.bpdecode.* (CPU + CUDA)
CMakeLists.txt    top-level build (scikit-build-core): csrc core + torch op
bindings/         HF / vLLM LogitsProcessor adapters (Phase 2)
bench/            throughput / TTFT / per-token overhead vs Outlines, XGrammar
tests/            differential tests vs the CPU reference; fuzzing
```

## CUDA modernization (baked into the phases)

- CSR / SoA transition tables, not AoS structs with fixed padding
- Multi-block grid, one warp per request; `__ballot_sync` to pack 32 tokens/word
- bf16 + log-domain (log-sum-exp) for the soft pass; Tensor Cores if factor
  updates batch into matmuls
- CUDA Graphs for the fixed-iteration lookahead loop
- Two streams: mask compute overlapped with the model forward pass
- Optional persistent constraint kernel to avoid per-token relaunch
- CMake + arch flags, streams/events, no deprecated APIs

## Phases

### Phase 0 -- scaffolding *(done)*

- [x] repo, packaging, CI, C++/CMake tree (CUDA optional)
- [x] regex -> code-point DFA (Thompson + subset construction)
- [x] token-level automaton (`TokenDFA`): lazy memoised `step` / `allowed` / `mask`
- [x] `Vocabulary` loading (`from_tokens`, `from_hf`; default model
      `Qwen/Qwen2.5-0.5B` -- GPT-2-style byte-level BPE, runs on a laptop)
- [x] `Constraint` interface + HF `LogitsProcessor` shim
- [x] CPU reference `RegexConstraint` + brute-force differential tests
- [x] C++ ABI (`FsaTable`, `TokenSymbols`) + scalar `compute_mask*` + gtests

### Phase 1 -- FSA path, single request, GPU mask *(done)*

CUDA kernels validated on an RTX 3090 (2026-09-10): `tests/test_cuda.py`
differential (22 cases) + `compute-sanitizer` memcheck, both clean. Rerun with
`scripts/gpu_check.sh`.

Host side -- runs without a GPU, covered by `pytest` + gtests:

- [x] byte-DFA x tokenizer product: the whole pipeline is on raw bytes. Regex
      `CharSet`s are lowered to their UTF-8 byte automaton (`regex/utf8.py`,
      exhaustively checked vs the platform codec); the DFA alphabet is 0..255;
      `TokenDFA` / `token_symbols` walk raw token bytes, so partial-UTF-8
      tokens from byte-level BPE resolve correctly. No more `surrogateescape`.
- [x] host FSA export: `bpdecode.fsa` flattens a compiled DFA + `Vocabulary`
      into `FsaTable` / `TokenSymbols` (the `csrc` ABI, POD arrays ready for
      CSR upload) plus scalar `step` / `compute_mask` / `advance_state_batch` /
      `apply_mask` mirrors of `mask_cpu.cpp`, differential-tested vs `TokenDFA`.
- [x] `build_reachability` ported to C++ (`mask_cpu.cpp`): boolean-BP backward
      reachability, so `FsaTable.live` is computed, not supplied. gtests.
- [x] `advance_state_batch` + fused `apply_mask_batch` (C++ scalar + gtests).
- [x] scikit-build-core builds `csrc` + the torch op library into the wheel;
      `torch.ops.bpdecode.{build_reachability,compute_mask,apply_mask_,advance_state}`
      via `bpdecode.ops` (`FsaTensors` bundle). CPU path differential-tested in
      `tests/test_ops.py`; CUDA path dispatched by tensor device.

CUDA kernels -- written, run only on GPU CI (`.github/workflows/gpu.yml`,
`scripts/gpu_check.sh`); `tests/test_cuda.py` is the CUDA-vs-CPU differential:

- [x] `compute_mask_batch_cuda`: one warp per request, `__ballot_sync` packs
      32 token verdicts per word.
- [x] `advance_state_batch_cuda` (thread/request); `apply_mask_batch_cuda`
      (warp/request, fused, no bitset round-trip).
- [x] `build_reachability_cuda`: iterative `live |= OR(succ)` fixpoint.
- [x] validated on an RTX 3090 -- differential + `compute-sanitizer` clean.
- [ ] nvbench microbench (deferred to Phase 5 perf work).

### Phase 2 -- batching + serving integration *(in progress)*

- [x] batched state array + continuous-batching lifecycle: `bpdecode.batch`
      (`ConstraintBatch` -- one int32 state per slot, `add`/`evict`/`reset`,
      whole-batch `apply_mask` / `commit` in one op call each).
- [x] shared compiled DFA across identical grammars: `GrammarCache` (LRU,
      pattern string -> one shared `FsaTensors`).
- [x] HF `LogitsProcessor` adapter: `bpdecode.hf.RegexLogitsProcessor`
      (per-row state, advances on the sampled token; greedy + sampling).
- [x] adaptive mask cache: `bpdecode.batch.MaskCache` -- LRU of per-state
      allow-sets (context-independent), stored as token-id lists so it stays
      kilobytes at 150k vocab. `ConstraintBatch(mask_cache=True)` -> a hit is a
      gather+scatter, no kernel launch.
- [x] dense token-transition table: `FsaTensors(dense=True)` precomputes
      `tok_next[state][token]` (`ops.build_token_transitions`, walk the vocab
      shortest-first). `apply_mask_` / `advance_state` then take a `tok_next`
      gather instead of re-walking token bytes -- 3-15x faster per step, beats
      `outlines_core` on non-trivial regexes (`bench/RESULTS.md`). `dense="auto"`
      (default) enables it when `num_states x vocab` <= ~32M.
- [x] vLLM adapter: `bpdecode.vllm` -- `RegexLogitsProcessor` (request-level,
      `SamplingParams(logits_processors=[...])`) + `RegexLogitsProcessorFactory`
      (grammar sharing); `BatchConstraintState` is the state machine for a V1
      batch-level `LogitsProcessor` (add/remove/move/advance/mask over
      `ConstraintBatch`). Tested with synthetic inputs; not yet run against a
      live vLLM engine.
- [x] benchmark vs Outlines: `bench/regex_mask.py` (compile time + per-token
      mask overhead + allowed-set cross-check). CPU run in `bench/RESULTS.md`
      -- output identical to `outlines_core`; compile now at parity (~17 ms
      after vectorising `token_symbols`); per-token mask still 2-15x slower on
      CPU (we recompute per step; Outlines has a precomputed token-FSM).
      GPU-batch rerun pending.
- [ ] end-to-end demo (Qwen2.5-0.5B) -- after the remaining phases.

### Phase 3 -- CFG / pushdown (JSON Schema, GBNF) *(in progress)*

- [x] GBNF front-end: `bpdecode.grammar` -- `parse_gbnf` -> `Grammar` IR
      (regex AST + `Ref`); `{m,n}` sugar expanded; `Grammar.is_regular()`.
- [x] rule NFAs: each rule -> byte-level NFA over `{byte-range, call(rule)}`
      edges (`grammar/nfa.py`, reuses the UTF-8 lowering).
- [x] config-set PDA (`grammar/pda.py`): runtime state is a set of stacks,
      epsilon-closure resolves call/return/eps to a fixpoint; per-rule
      co-reachability prune; depth + config-set caps.
- [x] `CFGConstraint` (`grammar/constraint.py`): CPU oracle, `accepts` /
      `advance` / `allowed_ids` by byte simulation. Validated against an
      independent recursive grammar matcher + brute-force token differential.
- [ ] JSON Schema -> `Grammar` (subset).
- [ ] persistent per-request execution stack; `compute_mask_pda` /
      `advance_state_pda` GPU kernels (push / pop, depth cap).
- [ ] context-dependent token split (context-independent mask precompute).
- [ ] benchmark vs XGrammar, llguidance; extra cross-check vs `lark`.

### Phase 4 -- soft lookahead (novel)

- weighted automaton; k-step backward sum-product -> per-(state, token) log-mass
- `soft_lookahead` kernel, CUDA-graph'd, bf16 log-domain
- expose as temperature-scaled logit bias
- eval: does soft guidance beat hard masking on structured-output accuracy
  (JSON-mode correctness, BFCL function-calling)? Report honestly, incl. a
  negative result.

### Phase 5 -- perf hardening & release

- stream overlap with forward pass, persistent kernel, Nsight tuning
- multi-GPU (shard by request -- state is tiny)
- docs, examples, `cibuildwheel` wheels, benchmark report

## Key risks

- **Tokenizer alignment** -- byte-level BPE, partial UTF-8, token healing.
  Dominant source of correctness bugs. Phase 1 moved the pipeline to bytes
  (regex -> UTF-8 byte automaton, tokens fed as raw bytes). Token healing is
  still open.
- **Token-level DFA blowup** for large vocab x complex grammar -> lazy
  construction + caching.
- **Moving target** -- XGrammar / llguidance improve fast; the soft-lookahead
  layer is the defensible differentiator.
- **Soft lookahead may not help** -- needs an honest eval gate before Phase 5.
