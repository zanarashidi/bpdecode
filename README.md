# bpdecode

GPU-accelerated **constrained decoding** for LLM inference: compute the
per-step token mask that keeps generation on a grammar-valid path, batched
across a serving workload, without stalling the GPU that runs the forward
pass.

The message-passing core grew out of a CUDA loopy belief-propagation project
(`loopy-belief-propagation-using-CUDA`) -- reachability in a constraint
automaton is boolean message passing to a fixpoint, and "soft" lookahead
guidance is the sum-product version of the same pass.

## Status: Phase 1 (FSA path) -- in progress

Phase 0 (host scaffolding) is complete; Phase 1 is lowering that pipeline onto
the GPU.

| Piece | Module | Notes |
|---|---|---|
| regex -> byte DFA | `bpdecode.regex` | Thompson NFA + subset construction; code-point classes lowered to a UTF-8 byte automaton (`regex/utf8.py`), alphabet is raw bytes |
| token-level automaton | `bpdecode.automaton` | lazy, memoised `step` / `allowed` / `mask` over a `Vocabulary`; consumes raw token bytes (partial-UTF-8 BPE tokens included) |
| vocabulary loading | `bpdecode.tokenizer` | `from_tokens` for tests; `from_hf` for byte-level BPE (defaults to `Qwen/Qwen2.5-0.5B`, laptop-friendly) |
| constraint interface | `bpdecode.interface` | `Constraint` protocol + HF `LogitsProcessor` shim |
| CPU reference | `bpdecode.reference` | `RegexConstraint` -- the correctness oracle for every later backend |
| FSA export | `bpdecode.fsa` | flatten DFA + vocab into `FsaTable` / `TokenSymbols` (the C++ ABI, ready for CSR upload) + a scalar `step` / `compute_mask` mirror |
| C++/CUDA core | `csrc/` | `build_reachability` (boolean-BP fixpoint), scalar `compute_mask*`; batched CUDA kernels (`compute_mask_batch_cuda`, `build_reachability_cuda`) written, not yet run on a GPU |

Still to land in Phase 1: `advance_state` + fused `apply_mask` kernels,
`scikit-build-core` wheel exposing `torch.ops.bpdecode.*`, and CUDA-vs-CPU
differential tests on GPU CI.

See [`docs/PLAN.md`](docs/PLAN.md) for the full roadmap (batching, pushdown/CFG,
sum-product lookahead, perf hardening).

## Quick start

```python
from bpdecode import RegexConstraint, Vocabulary

vocab = Vocabulary.from_tokens(["a", "b", "ab", "0", "1", "<eos>"], eos_id=5)
con = RegexConstraint(r"[01]+", vocab)

con.accepts(vocab.token_bytes.index(b"0"))   # -> True
scores = [0.0] * vocab.size
con.apply_(scores)                            # disallowed logits -> -inf
```

## Development

```bash
pip install -e '.[dev]'
pytest
ruff check .
```

The C++/CUDA tree under `csrc/` builds independently with CMake (CUDA optional)
and is not wired into the Python package yet:

```bash
cmake -S csrc -B build-cpp && cmake --build build-cpp && ctest --test-dir build-cpp
```

## License

MIT
