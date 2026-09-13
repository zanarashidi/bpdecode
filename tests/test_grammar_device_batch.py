"""CFGConstraintBatch / GrammarLogitsProcessor(backend="device") against the
CPU CFGConstraint oracle -- the on-device PDA kernel wired into the serving
path, exercised with real (multi-row, evict/reset) batch lifecycle traffic.
Everything here runs CPU-only (the kernels have a CPU implementation); the
CUDA path is the same op dispatched on a CUDA device, covered on the pod.
"""

from __future__ import annotations

import itertools
import random

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bpdecode.ops")

from bpdecode.grammar.constraint import CFGConstraint  # noqa: E402
from bpdecode.grammar.device import CFGConstraintBatch, build_pda_tensors  # noqa: E402
from bpdecode.grammar.gbnf import parse_gbnf  # noqa: E402
from bpdecode.grammar.pda import CompiledGrammar  # noqa: E402
from bpdecode.hf import GrammarLogitsProcessor  # noqa: E402
from bpdecode.tokenizer import Vocabulary  # noqa: E402

VOCAB = Vocabulary.from_tokens(
    ["(", ")", "x", "y", "{", "}", "a", "b", ":", "0", "1", ",", "<eos>"], eos_id=12
)
GRAMMARS = [
    'root ::= "(" root ")" | "x" | "y"',
    'root ::= "{" p ("," p)* "}"\np ::= [a-b] ":" [0-1]',
]


@pytest.mark.parametrize("src", GRAMMARS)
def test_batch_matches_cfgconstraint_random_walk(src: str) -> None:
    grammar = parse_gbnf(src)
    compiled = CompiledGrammar.build(grammar)
    g = build_pda_tensors(compiled, VOCAB)

    n = 3
    batch = CFGConstraintBatch(g, capacity=n)
    refs = [CFGConstraint.from_compiled(compiled, VOCAB) for _ in range(n)]
    for i in range(n):
        batch.add(i)

    rng = random.Random(0)
    for _ in range(12):
        logits = torch.zeros(n, VOCAB.size)
        batch.apply_mask(list(range(n)), logits)
        for i in range(n):
            got = {t for t in range(VOCAB.size) if logits[i, t].item() == 0.0}
            want = {t for t in range(VOCAB.size) if refs[i].accepts(t)}
            assert got == want, (src, i, got, want)
            assert batch.is_complete(i) == refs[i].is_complete()

        toks = []
        for i in range(n):
            allowed = sorted(t for t in range(VOCAB.size) if refs[i].accepts(t))
            toks.append(rng.choice(allowed))
        batch.commit(list(range(n)), torch.tensor(toks, dtype=torch.int32))
        for i in range(n):
            refs[i].advance(toks[i])


def test_batch_add_evict_reuses_slots() -> None:
    grammar = parse_gbnf('root ::= "x" | "y"')
    compiled = CompiledGrammar.build(grammar)
    g = build_pda_tensors(compiled, VOCAB)
    batch = CFGConstraintBatch(g, capacity=2)

    batch.add("a")
    batch.add("b")
    with pytest.raises(RuntimeError):
        batch.add("c")
    batch.evict("a")
    slot = batch.add("c")  # reuses a's freed slot
    assert slot in (0, 1)
    assert "a" not in batch and "b" in batch and "c" in batch


def test_broken_row_masks_everything_until_reset() -> None:
    grammar = parse_gbnf('root ::= "x"')
    compiled = CompiledGrammar.build(grammar)
    g = build_pda_tensors(compiled, VOCAB)
    batch = CFGConstraintBatch(g, capacity=1)
    batch.add(0)

    bad = VOCAB.token_bytes.index(b"y")
    status = batch.commit([0], torch.tensor([bad], dtype=torch.int32))
    assert int(status.item()) == -1
    assert batch.is_broken(0)

    logits = torch.zeros(1, VOCAB.size)
    batch.apply_mask([0], logits)
    assert torch.isneginf(logits).all()
    assert not batch.is_complete(0)

    batch.reset(0)
    assert not batch.is_broken(0)
    logits = torch.zeros(1, VOCAB.size)
    batch.apply_mask([0], logits)
    assert logits[0, VOCAB.token_bytes.index(b"x")].item() == 0.0


def test_logits_processor_device_backend_matches_cpu_backend() -> None:
    grammar = 'root ::= "(" root ")" | "x" | "y"'
    lp_cpu = GrammarLogitsProcessor(grammar, VOCAB, backend="cpu")
    lp_dev = GrammarLogitsProcessor(grammar, VOCAB, device="cpu", backend="device")

    input_ids = torch.zeros(2, 1, dtype=torch.long)
    rng = random.Random(1)
    for _ in range(8):
        s_cpu = torch.zeros(2, VOCAB.size)
        s_dev = torch.zeros(2, VOCAB.size)
        lp_cpu(input_ids, s_cpu)
        lp_dev(input_ids, s_dev)
        assert torch.equal(torch.isneginf(s_cpu), torch.isneginf(s_dev))
        assert lp_cpu.is_complete(0) == lp_dev.is_complete(0)
        assert lp_cpu.is_complete(1) == lp_dev.is_complete(1)

        toks = []
        for r in range(2):
            allowed = (~torch.isneginf(s_cpu[r])).nonzero().flatten().tolist()
            toks.append(rng.choice(allowed))
        input_ids = torch.cat(
            [input_ids, torch.tensor(toks, dtype=torch.long).unsqueeze(1)], dim=1
        )


def test_auto_backend_picks_device_off_cpu() -> None:
    lp = GrammarLogitsProcessor('root ::= "x"', VOCAB, device="cpu")
    assert lp.backend == "cpu"
    lp2 = GrammarLogitsProcessor('root ::= "x"', VOCAB, backend="device")
    assert lp2.backend == "device"


def test_exhaustive_short_strings_agree_via_batch() -> None:
    src = 'root ::= "x" | "y"'
    grammar = parse_gbnf(src)
    compiled = CompiledGrammar.build(grammar)
    g = build_pda_tensors(compiled, VOCAB)
    alphabet = [i for i in range(VOCAB.size) if i != VOCAB.eos_id]

    tested = 0
    for length in range(3):
        for combo in itertools.product(alphabet, repeat=length):
            batch = CFGConstraintBatch(g, capacity=1)
            ref = CFGConstraint.from_compiled(compiled, VOCAB)
            batch.add(0)
            dead = False
            for tok in combo:
                ok_ref = ref.accepts(tok)
                if ok_ref:
                    ref.advance(tok)
                status = batch.commit([0], torch.tensor([tok], dtype=torch.int32))
                ok_dev = int(status.item()) != -1
                assert ok_ref == ok_dev, (src, combo)
                if not ok_ref:
                    dead = True
                    break
            if dead:
                continue
            assert ref.is_complete() == batch.is_complete(0)
            tested += 1
    assert tested > 0
