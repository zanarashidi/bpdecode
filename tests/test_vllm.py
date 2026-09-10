"""vLLM adapters -- exercised with synthetic inputs (vLLM is not imported)."""

from __future__ import annotations

import re

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bpdecode.ops")

from bpdecode.reference import RegexConstraint  # noqa: E402
from bpdecode.tokenizer import Vocabulary  # noqa: E402
from bpdecode.vllm import (  # noqa: E402
    BatchConstraintState,
    RegexLogitsProcessor,
    RegexLogitsProcessorFactory,
)

TOKENS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", "-", "x", "</s>"]
VOCAB = Vocabulary.from_tokens(TOKENS, eos_id=len(TOKENS) - 1)
DECIMAL = "-?[0-9]+(\\.[0-9]+)?"


def _greedy(proc, bias, max_new=12):
    past: list[int] = []
    for _ in range(max_new):
        logits = bias.clone()
        logits = proc(past, logits)
        nxt = int(logits.argmax())
        if nxt == VOCAB.eos_id:
            break
        past.append(nxt)
    return past


def test_request_processor_output_is_a_valid_prefix() -> None:
    bias = torch.zeros(VOCAB.size)
    bias[TOKENS.index("x")] = 9.0  # push toward an always-illegal token
    proc = RegexLogitsProcessor(DECIMAL, VOCAB)
    past = _greedy(proc, bias)

    ref = RegexConstraint(DECIMAL, VOCAB)
    for tok in past:
        assert ref.accepts(tok)
        ref.advance(tok)
    assert TOKENS.index("x") not in past


def test_request_processor_tracks_reference() -> None:
    proc = RegexLogitsProcessor(DECIMAL, VOCAB)
    ref = RegexConstraint(DECIMAL, VOCAB)
    past: list[int] = []
    for tok in [TOKENS.index(c) for c in ("-", "4", ".", "2")]:
        logits = torch.zeros(VOCAB.size)
        proc(past, logits)
        got = {i for i in range(VOCAB.size) if logits[i].item() == 0.0}
        want = {i for i in range(VOCAB.size) if ref.accepts(i)}
        assert got == want
        ref.advance(tok)
        past.append(tok)
    proc(past, torch.zeros(VOCAB.size))  # consume the final token
    assert proc.is_complete


def test_factory_shares_compiled_grammar() -> None:
    factory = RegexLogitsProcessorFactory(VOCAB)
    p1 = factory.make(DECIMAL)
    p2 = factory.make(DECIMAL)
    assert p1._batch.fsa is p2._batch.fsa
    assert p1 is not p2


def test_batch_state_masks_only_its_rows() -> None:
    st = BatchConstraintState(VOCAB, mask_cache=False)
    st.add(0, "[0-9]+")
    st.add(2, "[0-9]+")
    # row 1 is some other request -- must be left alone
    logits = torch.zeros(3, VOCAB.size)
    logits[1, TOKENS.index("x")] = 3.0
    st.mask(logits)

    for r in (0, 2):
        assert torch.isneginf(logits[r, TOKENS.index("x")])
        assert logits[r, TOKENS.index("0")].item() == 0.0
    assert logits[1, TOKENS.index("x")].item() == 3.0  # untouched


def test_batch_state_lifecycle_matches_per_request() -> None:
    st = BatchConstraintState(VOCAB, mask_cache=True)
    refs = {r: RegexConstraint(DECIMAL, VOCAB) for r in (0, 1)}
    for r in refs:
        st.add(r, DECIMAL)

    plans = {
        0: [TOKENS.index(c) for c in ("1", "2", "3")],
        1: [TOKENS.index(c) for c in ("-", "9")],
    }
    gen: dict[int, list[int]] = {0: [], 1: []}
    for step in range(3):
        logits = torch.zeros(2, VOCAB.size)
        st.mask(logits)
        for r in (0, 1):
            got = {i for i in range(VOCAB.size) if logits[r, i].item() == 0.0}
            want = {i for i in range(VOCAB.size) if refs[r].accepts(i)}
            assert got == want, (r, step)
        for r in (0, 1):
            if step < len(plans[r]):
                tok = plans[r][step]
                refs[r].advance(tok)
                gen[r].append(tok)
                st.advance(r, gen[r])

    # evict row 0, move row 1 -> slot 0 (vLLM compaction), keep going
    st.remove(0)
    st.move(1, 0)
    logits = torch.zeros(1, VOCAB.size)
    st.mask(logits)
    want = {i for i in range(VOCAB.size) if refs[1].accepts(i)}
    got = {i for i in range(VOCAB.size) if logits[0, i].item() == 0.0}
    assert got == want


def test_batch_state_rejects_mixed_grammars() -> None:
    st = BatchConstraintState(VOCAB)
    st.add(0, "[0-9]+")
    with pytest.raises(NotImplementedError):
        st.add(1, "-?[0-9]+")


def test_greedy_toward_eos_completes_a_full_match() -> None:
    proc = RegexLogitsProcessor("[12][0-9][0-9]", VOCAB)
    bias = torch.zeros(VOCAB.size)
    bias[VOCAB.eos_id] = 9.0
    past = _greedy(proc, bias)
    assert re.fullmatch("[12][0-9][0-9]", "".join(TOKENS[i] for i in past))
