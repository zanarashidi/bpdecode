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
| FSA export | `bpdecode.fsa` | flatten DFA + vocab into `FsaTable` / `TokenSymbols` (the C++ ABI, ready for CSR upload) + scalar `step` / `compute_mask` / `advance_state` / `apply_mask` mirrors |
| C++/CUDA core | `csrc/` | `build_reachability` (boolean-BP fixpoint), scalar `compute_mask` / `advance_state` / fused `apply_mask`; matching batched CUDA kernels (warp-per-request, `__ballot_sync`) |
| torch ops | `bpdecode.ops` | `torch.ops.bpdecode.*` via `scikit-build-core`; CPU path tested, CUDA path dispatched by tensor device |

Still to land in Phase 1: run the CUDA-vs-CPU differential on a real GPU
(`scripts/gpu_check.sh` / the `gpu` workflow) to close it out.

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

Batched, on tensors (CPU now, CUDA when built on a GPU box):

```python
import torch
from bpdecode.ops import FsaTensors, apply_mask_

fsa = FsaTensors.build(r"[01]+", vocab)          # .to("cuda") to run on GPU
logits = torch.randn(batch, vocab.size)
apply_mask_(logits, fsa, states)                 # disallowed logits -> -inf, in place
```

## Development

The build compiles `csrc/` and the torch op extension via `scikit-build-core`,
so a C++ toolchain and `torch` are needed:

```bash
pip install torch            # or the CPU wheel: --index-url https://download.pytorch.org/whl/cpu
pip install scikit-build-core cmake ninja
pip install --no-build-isolation -e '.[dev]'
pytest
ruff check .
```

The `csrc/` tree also builds standalone with CMake (CUDA optional):

```bash
cmake -S csrc -B build-cpp && cmake --build build-cpp && ctest --test-dir build-cpp
```

CUDA kernels are validated on a GPU box with `scripts/gpu_check.sh` (or the
manual `gpu` GitHub workflow).

## License

MIT
