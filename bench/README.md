# Benchmarks

Results and analysis: [`RESULTS.md`](RESULTS.md).

## `regex_mask.py` -- bpdecode vs Outlines

```
pip install outlines transformers        # plus bpdecode, torch
python bench/regex_mask.py                # CPU
python bench/regex_mask.py --device cuda --batch 64   # on a GPU box
```

Grammar **compile** time (regex -> ready) and **per-token mask** overhead at a
given batch size, against
[`outlines_core`](https://github.com/dottxt-ai/outlines-core). A correctness
cross-check reports decode steps where the two libraries' allowed sets differ
(EOS excluded).

## `json_schema_mask.py` -- bpdecode vs XGrammar / llguidance

```
pip install xgrammar llguidance transformers
python bench/json_schema_mask.py
```

Same shape of measurement (compile + per-token mask, cold vs. warm) for
JSON-Schema-constrained decoding against
[XGrammar](https://github.com/mlc-ai/xgrammar) and
[llguidance](https://github.com/guidance-ai/llguidance).

## `soft_eval.py` / `soft_eval_modelweighted.py` / `lookahead_model_weighted.py` -- soft lookahead

Structured-output generation with `Qwen2.5-0.5B`, hard masking vs. the
automaton-count soft-lookahead bias (`bpdecode.ops.build_lookahead`, a
negative result) vs. a 1-step model-probability-weighted variant (works --
first as a single-request proof of concept in `soft_eval_modelweighted.py`,
then productionised as the batched, KV-cache-reusing
`bpdecode.lookahead.generate_model_weighted` (regex) /
`generate_model_weighted_cfg` (CFG / JSON Schema) in
`lookahead_model_weighted.py --grammar-mode regex|cfg`). See `RESULTS.md`
for the findings.

## Not yet built

- end-to-end tokens/sec with continuous batching + vLLM
- C++/CUDA microbenchmarks with [nvbench](https://github.com/NVIDIA/nvbench)
  (`-DBPDECODE_BUILD_BENCH=ON`, requires CUDA)
- on-device dense-table splicing for the PDA kernel (`grammar/regular.py`'s
  trick, ported to `grammar.device`) -- confirmed via a batch-scaling test
  as the real fix for why the CFG lookahead path doesn't speed up on GPU the
  way regex does (see `RESULTS.md`'s GPU section)
