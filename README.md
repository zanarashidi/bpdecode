# bpdecode

**Constrained decoding** for LLM inference: at every step, intersect the model's
next-token distribution with "which tokens keep generation on a grammar-valid
path" -- batched across a serving workload, on the GPU that holds the logits.

Regex, GBNF grammars, and a subset of JSON Schema. The automaton core grew out
of a CUDA loopy belief-propagation project: reachability in a constraint
automaton is boolean message-passing to a fixpoint, and the (experimental)
soft-lookahead pass is its sum-product version.

## Install

The wheel compiles `csrc/` and a small torch op extension, so a C++ toolchain
and `torch` are needed:

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

## How it works

```
pattern / grammar / JSON Schema
   │  regex.compile (Thompson NFA -> subset construction)
   │  grammar.gbnf / grammar.json_schema -> rule NFAs
   ▼
byte automaton            code points lowered to UTF-8 (regex/utf8.py); alphabet is raw bytes
   │  fsa.token_symbols  ×  the tokenizer's byte strings
   ▼
token-level table         tok_next[state][token]  (dense, or lazy for a CFG's pushdown)
   │
   ▼
per-step mask / advance    one kernel launch over the batch (torch.ops.bpdecode.*)
```

- **Regular** grammars (regex, and GBNF/JSON-Schema rules with no recursion)
  compile to a byte DFA and a dense `tok_next` table -- a decode step is a
  gather.
- **Context-free** grammars run a config-set pushdown automaton
  (`grammar.pda`); masks are memoised on the compiled grammar (keyed by
  config-set) and regular sub-loops (`json-char*` etc.) are spliced out to the
  dense path.
- **CUDA**: `csrc/` has the batched kernels (one warp per request,
  `__ballot_sync` token packing) for both paths, validated on an RTX 3090
  against the CPU reference (`scripts/gpu_check.sh`). The pushdown kernel
  (`bpdecode.grammar.device`) is a standalone device API -- not yet wired
  into `CFGConstraint` / `GrammarLogitsProcessor`, which use the (faster,
  once warm) CPU path today.

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

## Development

```bash
pip install --no-build-isolation -e '.[dev]'
pytest
ruff check .
cmake -S csrc -B build-cpp && cmake --build build-cpp && ctest --test-dir build-cpp
```

## License

MIT
