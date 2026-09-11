"""PDA CUDA kernels vs CPU, over the same grammars as test_pda_device.py.

Skipped unless a CUDA device is visible -- runs on GPU CI / scripts/gpu_check.sh.
"""

from __future__ import annotations

import itertools

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

pytest.importorskip("bpdecode.ops")

from bpdecode.grammar.device import (  # noqa: E402
    build_pda_tensors,
    pda_advance_state,
    pda_apply_mask_,
)
from bpdecode.grammar.gbnf import parse_gbnf  # noqa: E402
from bpdecode.grammar.pda import CompiledGrammar  # noqa: E402
from bpdecode.tokenizer import Vocabulary  # noqa: E402

VOCAB = Vocabulary.from_tokens(
    ["(", ")", "x", "y", "{", "}", "a", "b", ":", "0", "1", ",", "<eos>"], eos_id=12
)
GRAMMARS = [
    'root ::= "(" root ")" | "x" | "y"',
    'root ::= "{" p ("," p)* "}"\np ::= [a-b] ":" [0-1]',
]


@pytest.mark.parametrize("src", GRAMMARS)
def test_mask_and_advance_match_cpu(src: str) -> None:
    compiled = CompiledGrammar.build(parse_gbnf(src))
    cpu = build_pda_tensors(compiled, VOCAB)
    gpu = cpu.to("cuda")

    alphabet = [i for i in range(VOCAB.size) if i != VOCAB.eos_id]
    for length in range(3):
        for combo in itertools.product(alphabet, repeat=length):
            cfg_cpu = cpu.init_batch(1)
            cfg_gpu = gpu.init_batch(1)
            dead = False
            for tok in combo:
                t = torch.tensor([tok], dtype=torch.int32)
                ok_cpu = bool(pda_advance_state(cfg_cpu, cpu, t).item())
                ok_gpu = bool(pda_advance_state(cfg_gpu, gpu, t.cuda()).item())
                assert ok_cpu == ok_gpu, (src, combo)
                if not ok_cpu:
                    dead = True
                    break
            if dead:
                continue
            m_cpu = torch.zeros(1, VOCAB.size)
            m_gpu = torch.zeros(1, VOCAB.size, device="cuda")
            pda_apply_mask_(m_cpu, cpu, cfg_cpu)
            pda_apply_mask_(m_gpu, gpu, cfg_gpu)
            assert torch.equal(m_cpu, m_gpu.cpu()), (src, combo)
