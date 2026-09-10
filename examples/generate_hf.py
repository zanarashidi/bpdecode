"""End-to-end: constrain a Hugging Face model to a regex or a JSON Schema.

    pip install 'bpdecode[hf]'          # transformers + torch
    python examples/generate_hf.py

Runs Qwen2.5-0.5B unconstrained, then with a regex constraint, then with a
JSON-Schema constraint -- each dropped into ``model.generate`` as a
``LogitsProcessor``.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bpdecode.hf import GrammarLogitsProcessor, RegexLogitsProcessor

MODEL = "Qwen/Qwen2.5-0.5B"


def generate(model, tok, prompt: str, processors=None, max_new_tokens: int = 40) -> str:
    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(
            ids,
            logits_processor=processors or [],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True)


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()

    prompt = "The IPv4 address of the router is "
    print("prompt:", prompt)
    print("  unconstrained:", repr(generate(model, tok, prompt)))

    ipv4 = r"([0-9]{1,3}\.){3}[0-9]{1,3}"
    lp = RegexLogitsProcessor(ipv4, tok)
    print(f"  regex {ipv4!r}:", repr(generate(model, tok, prompt, [lp], 20)))

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "maxLength": 24},
            "paradigm": {"enum": ["systems", "functional", "object-oriented"]},
            "year": {"type": "integer"},
            "compiled": {"type": "boolean"},
        },
        "required": ["name", "paradigm", "year", "compiled"],
        "additionalProperties": False,
    }
    prompt = "Describe the Rust programming language as JSON. "
    print("\nprompt:", prompt)
    print("  unconstrained:", repr(generate(model, tok, prompt)))
    gp = GrammarLogitsProcessor.from_json_schema(schema, tok)
    print("  JSON Schema  :", repr(generate(model, tok, prompt, [gp], 80)))


if __name__ == "__main__":
    main()
