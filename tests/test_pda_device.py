"""torch.ops.bpdecode.pda_* -- the device-facing PDA -- against the Python
config-set PDA (grammar.pda.PDA) over the same grammars test_cfg.py uses.

The CUDA kernels share this file's grammars for the GPU differential
(tests/test_cuda.py-style), run on a pod; here everything runs CPU-only.
"""

from __future__ import annotations

import itertools

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bpdecode.ops")

from bpdecode.grammar.device import (  # noqa: E402
    PDA_CONFIG_FLAT,
    build_pda_tensors,
    pda_advance_state,
    pda_apply_mask_,
)
from bpdecode.grammar.gbnf import parse_gbnf  # noqa: E402
from bpdecode.grammar.pda import PDA, CompiledGrammar  # noqa: E402
from bpdecode.tokenizer import Vocabulary  # noqa: E402

VOCAB = Vocabulary.from_tokens(
    ["(", ")", "x", "y", "{", "}", "a", "b", ":", "0", "1", ",", "<eos>"], eos_id=12
)
GRAMMARS = [
    'root ::= "(" root ")" | "x" | "y"',
    'root ::= "{" p ("," p)* "}"\np ::= [a-b] ":" [0-1]',
    'root ::= "x" | "y"',
]


def _allowed_via_ops(g_tensors, cfg) -> set[int]:
    logits = torch.zeros(1, VOCAB.size)
    pda_apply_mask_(logits, g_tensors, cfg)
    return {i for i in range(VOCAB.size) if logits[0, i].item() == 0.0}


def _allowed_via_reference(pda: PDA, vocab: Vocabulary) -> set[int]:
    out = set()
    for t in range(vocab.size):
        if t == vocab.eos_id:
            if pda.is_complete():
                out.add(t)
            continue
        snap = pda.configs
        ok = True
        for b in vocab.token_bytes[t]:
            if not pda.advance_byte(b):
                ok = False
                break
        pda.set_configs(snap)
        if ok:
            out.add(t)
    return out


@pytest.mark.parametrize("src", GRAMMARS)
def test_random_walk_matches_reference(src: str) -> None:
    grammar = parse_gbnf(src)
    compiled = CompiledGrammar.build(grammar)
    g_tensors = build_pda_tensors(compiled, VOCAB)

    pda = PDA(compiled)
    cfg = g_tensors.init_batch(1)

    import random

    rng = random.Random(0)
    for _ in range(15):
        want = _allowed_via_reference(pda, VOCAB)
        got = _allowed_via_ops(g_tensors, cfg)
        assert got == want, (src, want, got)
        if not want - {VOCAB.eos_id}:
            break
        choices = sorted(want - {VOCAB.eos_id}) or sorted(want)
        tok = rng.choice(choices)
        ok_ref = pda.advance_byte(VOCAB.token_bytes[tok][0]) if len(
            VOCAB.token_bytes[tok]
        ) == 1 else all(pda.advance_byte(b) for b in VOCAB.token_bytes[tok])
        assert ok_ref
        ok_dev = pda_advance_state(cfg, g_tensors, torch.tensor([tok], dtype=torch.int32))
        assert bool(ok_dev.item())


@pytest.mark.parametrize("src", GRAMMARS)
def test_exhaustive_short_strings_agree(src: str) -> None:
    grammar = parse_gbnf(src)
    compiled = CompiledGrammar.build(grammar)
    g_tensors = build_pda_tensors(compiled, VOCAB)
    alphabet = [i for i in range(VOCAB.size) if i != VOCAB.eos_id]

    tested = 0
    for length in range(4):
        for combo in itertools.product(alphabet, repeat=length):
            pda = PDA(compiled)
            cfg = g_tensors.init_batch(1)
            dead = False
            for tok in combo:
                ok_ref = all(pda.advance_byte(b) for b in VOCAB.token_bytes[tok])
                ok_dev = bool(
                    pda_advance_state(
                        cfg, g_tensors, torch.tensor([tok], dtype=torch.int32)
                    ).item()
                )
                assert ok_ref == ok_dev, (src, combo)
                if not ok_ref:
                    dead = True
                    break
            if dead:
                continue
            assert pda.is_complete() == (
                _allowed_via_ops(g_tensors, cfg).__contains__(VOCAB.eos_id)
            )
            tested += 1
    assert tested > 0


def test_advance_state_rejects_and_leaves_config_unchanged() -> None:
    grammar = parse_gbnf('root ::= "x" | "y"')
    compiled = CompiledGrammar.build(grammar)
    g_tensors = build_pda_tensors(compiled, VOCAB)
    cfg = g_tensors.init_batch(1)
    before = cfg.clone()

    bad = torch.tensor([VOCAB.token_bytes.index(b"(")], dtype=torch.int32)
    ok = pda_advance_state(cfg, g_tensors, bad)
    assert not bool(ok.item())
    assert torch.equal(cfg, before)


def test_batch_rows_independent() -> None:
    grammar = parse_gbnf('root ::= "(" root ")" | "x"')
    compiled = CompiledGrammar.build(grammar)
    g_tensors = build_pda_tensors(compiled, VOCAB)
    cfg = g_tensors.init_batch(3)

    p, x = VOCAB.token_bytes.index(b"("), VOCAB.token_bytes.index(b"x")
    toks = torch.tensor([p, x, p], dtype=torch.int32)
    ok = pda_advance_state(cfg, g_tensors, toks)
    assert ok.tolist() == [1, 1, 1]

    logits = torch.zeros(3, VOCAB.size)
    pda_apply_mask_(logits, g_tensors, cfg)
    # row 1 (took 'x') is complete: eos allowed, '(' not
    assert logits[1, VOCAB.eos_id].item() == 0.0
    # rows 0 and 2 (took '(') are mid-recursion: '(' and 'x' allowed, eos not
    for r in (0, 2):
        assert logits[r, VOCAB.eos_id].item() == float("-inf")
        assert logits[r, VOCAB.token_bytes.index(b"(")].item() == 0.0


def test_config_flat_size_matches_native() -> None:
    grammar = parse_gbnf('root ::= "x"')
    compiled = CompiledGrammar.build(grammar)
    g_tensors = build_pda_tensors(compiled, VOCAB)
    cfg = g_tensors.init_config()
    assert cfg.shape == (PDA_CONFIG_FLAT,)
