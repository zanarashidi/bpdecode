"""CUDA kernels vs the CPU reference, over a regex suite.

Skipped unless a CUDA device is visible. This is the differential that closes
the Phase 1 "CUDA vs CPU" checkbox; it runs on GPU CI
(`.github/workflows/gpu.yml`), not on a laptop.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

ops = pytest.importorskip("bpdecode.ops")

from bpdecode.tokenizer import Vocabulary  # noqa: E402

FsaTensors = ops.FsaTensors

VOCAB = Vocabulary.from_tokens(
    ["a", "b", "c", "ab", "ba", "0", "1", "12", "-", ".", "aa", "café", "é", "<eos>"],
    eos_id=13,
)
PATTERNS = [
    "[01]+",
    "a+b*",
    "(ab)+",
    "-?[0-9]+(\\.[0-9]+)?",
    "a|bc|12",
    "[a-c]*",
    "caf(e|é)",
]


def _states(dfa) -> list[int]:
    seen, stack = {dfa.start}, [dfa.start]
    while stack:
        for t in dfa.trans[stack.pop()]:
            if t not in seen:
                seen.add(t)
                stack.append(t)
    return sorted(seen)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_apply_mask_cuda_matches_cpu(pattern: str) -> None:
    from bpdecode.regex import compile_regex

    dfa = compile_regex(pattern)
    states = _states(dfa)
    st = torch.tensor(states, dtype=torch.int32)

    cpu = FsaTensors.build(pattern, VOCAB, dense=False)
    gpu = cpu.to("cuda")

    base = torch.randn(len(states), VOCAB.size)
    got_cpu = base.clone()
    ops.apply_mask_(got_cpu, cpu, st)
    got_gpu = base.cuda()
    ops.apply_mask_(got_gpu, gpu, st.cuda())

    assert torch.equal(got_cpu, got_gpu.cpu())

    # dense path (pure torch) on CUDA must agree with the kernel
    dense_gpu = base.cuda()
    ops.apply_mask_(dense_gpu, gpu.densify(), st.cuda())
    assert torch.equal(got_cpu, dense_gpu.cpu())


@pytest.mark.parametrize("pattern", PATTERNS)
def test_compute_mask_cuda_matches_cpu(pattern: str) -> None:
    from bpdecode.regex import compile_regex

    dfa = compile_regex(pattern)
    st = torch.tensor(_states(dfa), dtype=torch.int32)
    cpu = FsaTensors.build(pattern, VOCAB, dense=False)
    gpu = cpu.to("cuda")
    assert torch.equal(
        ops.compute_mask(cpu, st), ops.compute_mask(gpu, st.cuda()).cpu()
    )


@pytest.mark.parametrize("pattern", PATTERNS)
def test_advance_state_cuda_matches_cpu(pattern: str) -> None:
    from bpdecode.regex import compile_regex

    dfa = compile_regex(pattern)
    states, tokens = [], []
    for s in _states(dfa):
        for tid in range(VOCAB.size):
            states.append(s)
            tokens.append(tid)
    st = torch.tensor(states, dtype=torch.int32)
    tk = torch.tensor(tokens, dtype=torch.int32)

    cpu = FsaTensors.build(pattern, VOCAB, dense=False)
    gpu = cpu.to("cuda")

    got_cpu = ops.advance_state(cpu, st, tk)
    got_gpu = ops.advance_state(gpu, st.cuda(), tk.cuda())
    assert torch.equal(got_cpu, got_gpu.cpu())


def test_build_reachability_cuda_matches_cpu() -> None:
    from bpdecode.fsa import build_reachability

    bundle = FsaTensors.build("-?[0-9]+(\\.[0-9]+)?", VOCAB, dense=False)
    ref = build_reachability(
        len(bundle.accept),
        bundle.num_symbols,
        bundle.trans.tolist(),
        bundle.accept.tolist(),
    )
    # host op path
    got = torch.ops.bpdecode.build_reachability(
        bundle.trans, bundle.accept, bundle.num_symbols
    )
    assert got.tolist() == ref
