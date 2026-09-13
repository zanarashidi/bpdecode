# bpdecode

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-optional-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-yellow)

> **Status: pre-alpha / research code.** APIs may change without notice; this
> has not been used in production and hasn't had independent review.

**Constrained decoding** for LLM inference: at every step, intersect the model's
next-token distribution with "which tokens keep generation on a grammar-valid
path" -- batched across a serving workload, on the GPU that holds the logits.

Regex, GBNF grammars, and a subset of JSON Schema. The automaton core grew out
of a CUDA loopy belief-propagation project: reachability in a constraint
automaton is boolean message-passing to a fixpoint, and the (experimental)
soft-lookahead pass is its sum-product version.

## Contents

- [Install](#install)
- [Use it](#use-it)
- [How it works](#how-it-works)
- [Why this instead of Outlines / XGrammar / vLLM's built-in guided decoding](#why-this-instead-of-outlines--xgrammar--vllms-built-in-guided-decoding)
- [Benchmarks](#benchmarks)
- [Limitations](#limitations)
- [Development](#development)
- [License](#license)

## Install

Requirements: Python 3.10+, `torch` >= 2.2, and a C++20 toolchain (the wheel
compiles `csrc/` and a small torch op extension at install time). CUDA is
optional -- everything runs CPU-only, including on a laptop; the kernels build
in automatically if `nvcc` is found.

```bash
pip install torch                    # CPU: --index-url https://download.pytorch.org/whl/cpu
pip install scikit-build-core cmake ninja
pip install --no-build-isolation -e '.[hf]'
```

## Use it

**Regex-constrain a Hugging Face model:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from bpdecode.hf import RegexLogitsProcessor

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")

lp = RegexLogitsProcessor(r"([0-9]{1,3}\.){3}[0-9]{1,3}", tok)
ids = tok("The router IP is ", return_tensors="pt").input_ids
print(tok.decode(model.generate(ids, logits_processor=[lp], max_new_tokens=20)[0]))
# ... 192.168.1.100
```

**JSON Schema:**

```python
from bpdecode.hf import GrammarLogitsProcessor

schema = {
    "type": "object",
    "properties": {"name": {"type": "string", "maxLength": 24},
                   "year": {"type": "integer"},
                   "compiled": {"type": "boolean"}},
    "required": ["name", "year", "compiled"],
    "additionalProperties": False,
}
gp = GrammarLogitsProcessor.from_json_schema(schema, tok)
# -> {"name": "Rust", "year": 2010, "compiled": true}
```

**vLLM:** `from bpdecode.vllm import RegexLogitsProcessorFactory` -- pass
`SamplingParams(logits_processors=[factory.make(pattern)])`.

**Batched, on tensors** (the serving path -- CPU or CUDA by tensor device):

```python
from bpdecode.batch import GrammarCache, ConstraintBatch

cache = GrammarCache(vocab)                       # compile-once, shared across requests
batch = ConstraintBatch(cache.get(pattern), capacity=256, device="cuda")
batch.add(request_id)                             # continuous batching: add / evict / reset
batch.apply_mask(active_ids, logits)             # one call masks the whole batch
batch.commit(active_ids, sampled_tokens)         # one call advances it
```

See [`examples/`](examples/).

**No model, no tensors:** `RegexConstraint` is the plain-Python automaton --
it never touches a tensor at runtime, even though the package install still
needs `torch` present. Useful for testing or embedding the constraint logic
somewhere else:

```python
from bpdecode import RegexConstraint, Vocabulary

vocab = Vocabulary.from_tokens(["a", "b", "ab", "0", "1", "<eos>"], eos_id=5)
con = RegexConstraint(r"[01]+", vocab)
con.accepts(vocab.token_bytes.index(b"0"))   # -> True
```

## How it works

![how it works](assets/how-it-works-light.svg#gh-light-mode-only)
![how it works](assets/how-it-works-dark.svg#gh-dark-mode-only)

- **Regular** grammars (regex, and GBNF/JSON-Schema rules with no recursion)
  compile to a byte DFA and a dense `tok_next` table -- a decode step is a
  gather.
- **Context-free** grammars run a config-set pushdown automaton
  (`grammar.pda`); masks are memoised on the compiled grammar (keyed by
  config-set) and regular sub-loops (`json-char*` etc.) are spliced out to the
  dense path.
- **CUDA**: `csrc/` has the batched kernels (one warp per request,
  `__ballot_sync` token packing) for both paths, validated on an RTX 3090
  against the CPU reference (`scripts/gpu_check.sh`). `GrammarLogitsProcessor`
  picks between two backends: `"cpu"` (per-row `CFGConstraint`, masks
  memoised on the compiled grammar -- fastest once warm) and `"device"`
  (`CFGConstraintBatch`, the on-device PDA kernel -- one `torch.ops.bpdecode.pda_*`
  launch masks/advances the whole batch, no per-row Python or memo, CPU or
  CUDA). `backend="auto"` (default) follows `device`.

## Why this instead of Outlines / XGrammar / vLLM's built-in guided decoding

Outlines, XGrammar, and llguidance are mature, well-tested libraries doing the
same core job (constrained decoding via a Rust/C++ FSM), and for most use
cases they're the safer choice today given this project's status above. Two
reasons to reach for bpdecode instead:

- it's **batched natively** -- `ConstraintBatch` masks/advances an entire
  serving batch in one kernel call, with a shared compiled grammar and an
  adaptive mask cache, rather than one matcher object per request; and once a
  grammar is warm it's faster per token than any of the three (see below).
- the **soft-lookahead angle** -- steering the model away from
  valid-but-dead-end tokens instead of only hard-masking, the same idea as
  "expected future grammaticality" in
  [Grammar-Aligned Decoding](https://arxiv.org/abs/2405.21047) -- isn't
  something Outlines/XGrammar/llguidance ship. bpdecode's own count-based
  version of it doesn't work (a documented negative result); a
  model-probability-weighted version does, at the cost of extra forward
  passes. It's not novel research -- see GAD and
  [Constrained Decoding with Speculative Lookaheads](https://arxiv.org/abs/2412.10418)
  for the theory -- but it isn't in the three libraries above's production
  APIs either.

## Benchmarks

`bench/RESULTS.md`. Against `outlines_core`, `xgrammar`, `llguidance` on a
151k-token vocab:

- **regex**, per-token mask: ~8 µs on GPU (flat across patterns, beats
  `outlines_core` on realistic ones); ~90-140 µs on CPU.
- **JSON Schema**, warm: **~1 µs** -- an order of magnitude under xgrammar,
  ~50x under llguidance. The cost is a one-time per-grammar warmup.
- **soft lookahead**: negative result, recorded honestly -- count-based
  continuation weighting over-extends and doesn't beat hard masking. A 1-step
  model-probability-weighted variant fixes it (`bench/soft_eval_modelweighted.py`)
  at the cost of extra forward passes per decode step; not productionised.

CUDA kernels validated on an RTX 3090 (`scripts/gpu_check.sh`): differential
vs the CPU reference + `compute-sanitizer`, clean.

## Limitations

- **Regex:** no anchors, backreferences, or lookaround.
- **JSON Schema:** objects emit properties in schema order -- every string
  produced validates, but not every valid property ordering is producible.
  `minimum` / `maximum` on numbers aren't enforced (not expressible as a
  grammar).
- **CFG / pushdown:** the config-set is bounded (8 alternative stacks x 32
  frames deep) to keep it a fixed size. A pathologically ambiguous or deeply
  recursive grammar silently saturates rather than raising -- fine for
  JSON-Schema-scale grammars, not verified beyond that.
- **Soft lookahead:** the count-based version (`build_lookahead` /
  `apply_soft_`) is a documented negative result, not a recommended feature --
  see the Benchmarks section. The model-probability-weighted variant that does
  work is an unshipped experiment (`bench/soft_eval_modelweighted.py`).
- The on-device PDA kernel (`bpdecode.grammar.device`) exists and is
  GPU-validated but isn't wired into `CFGConstraint` / `GrammarLogitsProcessor`
  yet -- see "How it works" above.

## Development

```bash
pip install --no-build-isolation -e '.[dev]'
pytest
ruff check .
cmake -S csrc -B build-cpp && cmake --build build-cpp && ctest --test-dir build-cpp
```

## License

MIT
