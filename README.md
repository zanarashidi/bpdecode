# bpdecode

GPU-accelerated **constrained decoding** for LLM inference: compute the
per-step token mask that keeps generation on a grammar-valid path, batched
across a serving workload, without stalling the GPU that runs the forward
pass.

The message-passing core grew out of a CUDA loopy belief-propagation project
(`loopy-belief-propagation-using-CUDA`) -- reachability in a constraint
automaton is boolean message passing to a fixpoint, and "soft" lookahead
guidance is the sum-product version of the same pass.

## Status: Phase 0 (scaffolding)

Implemented so far -- **host side only, no CUDA yet**:

| Piece | Module | Notes |
|---|---|---|
| regex -> code-point DFA | `bpdecode.regex` | Thompson NFA + subset construction; symbol-class transitions |
| token-level automaton | `bpdecode.automaton` | lazy, memoised `step` / `allowed` / `mask` over a `Vocabulary` |
| vocabulary loading | `bpdecode.tokenizer` | `from_tokens` for tests; `from_hf` for byte-level BPE (defaults to `Qwen/Qwen2.5-0.5B`, laptop-friendly) |
| constraint interface | `bpdecode.interface` | `Constraint` protocol + HF `LogitsProcessor` shim |
| CPU reference | `bpdecode.reference` | `RegexConstraint` -- the correctness oracle for every later backend |

See [`docs/PLAN.md`](docs/PLAN.md) for the full roadmap (FSA kernels, batching,
pushdown/CFG, sum-product lookahead, perf hardening).

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

The C++/CUDA tree under `csrc/` builds independently with CMake and is not
wired into the Python package yet (Phase 1):

```bash
cmake -S csrc -B build-cpp && cmake --build build-cpp && ctest --test-dir build-cpp
```

## License

MIT
