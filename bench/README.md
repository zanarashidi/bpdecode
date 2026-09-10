# Benchmarks

## `regex_mask.py` -- bpdecode vs Outlines

```
pip install outlines transformers        # plus bpdecode, torch
python bench/regex_mask.py                # CPU
python bench/regex_mask.py --device cuda --batch 64   # on a GPU box
```

Measures grammar **compile** time (regex -> ready) and **per-token mask**
overhead at a given batch size, against
[`outlines_core`](https://github.com/dottxt-ai/outlines-core).  A correctness
cross-check reports decode steps where the two libraries' allowed sets differ
(EOS excluded).  See [`RESULTS.md`](RESULTS.md) for a recorded run and analysis.

## Planned

- vs [XGrammar](https://github.com/mlc-ai/xgrammar) /
  [llguidance](https://github.com/guidance-ai/llguidance) (Phase 3, CFG)
- end-to-end tokens/sec with continuous batching + vLLM (Phase 2 demo)
- Phase 4: structured-output task accuracy, hard mask vs soft lookahead
- C++/CUDA microbenchmarks with [nvbench](https://github.com/NVIDIA/nvbench)
  (`-DBPDECODE_BUILD_BENCH=ON`, requires CUDA)
