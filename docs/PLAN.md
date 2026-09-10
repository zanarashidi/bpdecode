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
                              (+ scalar step/compute_mask mirror of the C++ core)
csrc/             C++/CUDA core: FsaTable/TokenSymbols ABI, mask kernels
  include/bpdecode/mask.hpp    the ABI callers compile against
  src/mask_cpu.cpp             scalar reference + build_reachability
  src/mask_cuda.cu             batched kernels (warp/request, __ballot_sync)
bindings/         torch custom op + HF / vLLM LogitsProcessor adapters (Phase 2)
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

### Phase 1 -- FSA path, single request, GPU mask *(in progress)*

Done -- host side, runs without a GPU:

- [x] byte-DFA x tokenizer product: the whole pipeline is on raw bytes. Regex
      `CharSet`s are lowered to their UTF-8 byte automaton (`regex/utf8.py`,
      exhaustively checked vs the platform codec); the DFA alphabet is 0..255;
      `TokenDFA` / `token_symbols` walk raw token bytes, so partial-UTF-8
      tokens from byte-level BPE resolve correctly. No more `surrogateescape`.
- [x] host FSA export: `bpdecode.fsa` flattens a compiled DFA + `Vocabulary`
      into `FsaTable` / `TokenSymbols` (the `csrc` ABI, POD arrays ready for
      CSR upload) plus a scalar `step` / `compute_mask` mirror of
      `mask_cpu.cpp`, differential-tested against `TokenDFA`.
- [x] `build_reachability` ported to C++ (`mask_cpu.cpp`): boolean-BP backward
      reachability, so `FsaTable.live` is computed, not supplied. gtests.

Written, not yet exercised (no local GPU -- needs GPU CI):

- [x] `compute_mask_batch_cuda`: one warp per request, `__ballot_sync` packs
      32 token verdicts per word, caller-supplied stream.
- [x] `build_reachability_cuda`: iterative `live |= OR(succ)` fixpoint.

Still to do:

- [ ] `advance_state` kernel; fused `apply_mask` (Triton/CUDA)
- [ ] scikit-build-core: compile `csrc` into the wheel; `torch.ops.bpdecode.*`
- [ ] correctness: run the CUDA path vs the CPU reference over a regex suite
      on GPU CI; wire an nvbench microbench

### Phase 2 -- batching + serving integration

- batched state array, shared compiled DFA across identical grammars
- continuous-batching add / evict / reset
- device-side adaptive mask cache (context-independent masks, LRU)
- HF + vLLM `LogitsProcessor` adapters; end-to-end generation demo
- benchmark vs Outlines

### Phase 3 -- CFG / pushdown (JSON Schema, GBNF)

- EBNF -> PDA compiler; JSON Schema -> EBNF
- persistent per-request execution stack in global memory (depth cap)
- `compute_mask_pda` / `advance_state_pda` (push / pop)
- context-dependent tokens (one BPE token spanning multiple terminals)
- correctness vs `lark`; benchmark vs XGrammar, llguidance

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
