# Benchmarks

Placeholder for Phase 2+. Planned comparisons against
[Outlines](https://github.com/dottxt-ai/outlines),
[XGrammar](https://github.com/mlc-ai/xgrammar) and
[llguidance](https://github.com/guidance-ai/llguidance):

- per-step mask compute latency vs vocab size and grammar complexity
- end-to-end tokens/sec with continuous batching (vLLM integration)
- time-to-first-token overhead from grammar compilation
- Phase 4: structured-output task accuracy, hard mask vs soft lookahead

C++/CUDA microbenchmarks will use [nvbench](https://github.com/NVIDIA/nvbench)
(enable with `-DBPDECODE_BUILD_BENCH=ON`, requires CUDA).
