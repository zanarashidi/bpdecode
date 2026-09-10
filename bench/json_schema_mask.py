"""bpdecode vs XGrammar vs llguidance on JSON-Schema-constrained decoding.

    pip install xgrammar llguidance transformers      # plus bpdecode
    python bench/json_schema_mask.py

Measures, per schema:

* compile   -- schema -> ready-to-mask
* cold      -- one full generation on a fresh matcher (first request, cache warmup)
* warm      -- a second full generation (steady state)

`bpdecode` also pays a one-time per-vocabulary token-trie build (~0.4 s),
reported once. All CPU; times are per token per sequence unless noted.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

SCHEMAS: dict[str, dict] = {
    "person": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "email": {"type": "string"},
        },
        "required": ["name", "age", "email"],
        "additionalProperties": False,
    },
    "nested": {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "meta": {
                "type": "object",
                "properties": {"tag": {"type": "string"}, "score": {"type": "number"}},
                "required": ["tag", "score"],
                "additionalProperties": False,
            },
        },
        "required": ["id", "meta"],
        "additionalProperties": False,
    },
}
SAMPLES: dict[str, str] = {
    "person": '{"name": "Ada Lovelace", "age": 36, "email": "ada@example.org"}',
    "nested": '{"id": 7, "meta": {"tag": "alpha", "score": 9.5}}',
}


@dataclass
class Row:
    name: str
    compile_ms: dict[str, float]
    cold_us: dict[str, float]
    warm_us: dict[str, float]


def _median(fn, n=5) -> float:
    xs = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        xs.append(time.perf_counter() - t)
    xs.sort()
    return xs[len(xs) // 2]


def run(model: str) -> list[Row]:
    from transformers import AutoTokenizer

    hf = AutoTokenizer.from_pretrained(model)
    rows: list[Row] = []

    import xgrammar as xgr

    from bpdecode.grammar import CFGConstraint, json_schema_to_grammar
    from bpdecode.grammar.pda import CompiledGrammar
    from bpdecode.grammar.tokentrie import token_trie
    from bpdecode.tokenizer import Vocabulary

    bp_vocab = Vocabulary.from_hf(hf)
    t = time.perf_counter()
    token_trie(bp_vocab)
    trie_ms = (time.perf_counter() - t) * 1e3

    xtok = xgr.TokenizerInfo.from_huggingface(hf)
    xcomp = xgr.GrammarCompiler(xtok)

    llt = grm_compiler = None
    try:
        import llguidance
        import llguidance.hf

        llt = llguidance.hf.from_tokenizer(hf)
        grm_compiler = llguidance
    except Exception as exc:  # pragma: no cover
        print(f"(llguidance unavailable: {exc})")

    def bench_schema(name: str, schema: dict) -> Row:
        js = json.dumps(schema)
        ids = [int(x) for x in hf(SAMPLES[name], add_special_tokens=False).input_ids]
        n = len(ids)
        compile_ms: dict[str, float] = {}
        cold_us: dict[str, float] = {}
        warm_us: dict[str, float] = {}

        compile_ms["bp"] = _median(
            lambda: CompiledGrammar.build(json_schema_to_grammar(schema))
        ) * 1e3
        comp = CompiledGrammar.build(json_schema_to_grammar(schema))

        def bp_gen() -> None:
            con = CFGConstraint.from_compiled(comp, bp_vocab)
            for i in ids:
                con.allowed_ids()
                con.advance(i)

        cold_us["bp"] = _median(bp_gen, 1) / n * 1e6
        warm_us["bp"] = _median(bp_gen, 5) / n * 1e6

        compile_ms["xgr"] = _median(lambda: xcomp.compile_json_schema(js)) * 1e3
        xcg = xcomp.compile_json_schema(js)
        mask = xgr.allocate_token_bitmask(1, xtok.vocab_size)

        def xgr_gen() -> None:
            m = xgr.GrammarMatcher(xcg)
            for i in ids:
                m.fill_next_token_bitmask(mask)
                m.accept_token(i)

        cold_us["xgr"] = _median(xgr_gen, 1) / n * 1e6
        warm_us["xgr"] = _median(xgr_gen, 5) / n * 1e6

        if llt is not None:
            compile_ms["llg"] = _median(
                lambda: grm_compiler.JsonCompiler().compile(js)
            ) * 1e3
            g = grm_compiler.JsonCompiler().compile(js)

            def llg_gen() -> None:
                m = grm_compiler.LLMatcher(llt, g)
                for i in ids:
                    m.compute_bitmask()
                    m.consume_token(i)

            cold_us["llg"] = _median(llg_gen, 1) / n * 1e6
            warm_us["llg"] = _median(llg_gen, 5) / n * 1e6

        return Row(name, compile_ms, cold_us, warm_us)

    rows = [bench_schema(n, s) for n, s in SCHEMAS.items()]
    print(f"\nbpdecode token-trie build: {trie_ms:.0f} ms (once per vocabulary)\n")
    return rows


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    args = ap.parse_args()

    rows = run(args.model)
    libs = ["bp", "xgr", "llg"]
    hdr = f"{'schema':<10} {'metric':<10}" + "".join(f"{lib:>12}" for lib in libs)
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        for metric, d, unit in (
            ("compile", r.compile_ms, "ms"),
            ("cold", r.cold_us, "us"),
            ("warm", r.warm_us, "us"),
        ):
            cells = "".join(
                f"{d[lib]:>10.1f}{unit}" if lib in d else f"{'-':>12}" for lib in libs
            )
            print(f"{r.name:<10} {metric:<10}{cells}")
        print()
    print("compile = schema -> ready (ms).  cold/warm = per token per sequence (us).")


if __name__ == "__main__":
    main()
