"""`utf8_sequences` must cover exactly the scalars in a range, byte for byte."""

from __future__ import annotations

import pytest

from bpdecode.regex.utf8 import UNICODE_MAX, utf8_sequences


def _matches(seqs, encoded: bytes) -> bool:
    for seq in seqs:
        if len(seq) == len(encoded) and all(
            lo <= b <= hi for (lo, hi), b in zip(seq, encoded, strict=True)
        ):
            return True
    return False


def _brute_check(lo: int, hi: int) -> None:
    seqs = utf8_sequences(lo, hi)
    for cp in range(lo, hi + 1):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        enc = chr(cp).encode("utf-8")
        assert _matches(seqs, enc), f"U+{cp:04X} missing from utf8_sequences({lo:#x},{hi:#x})"
    # nothing outside the range leaks in (sample the neighbourhood)
    below = range(max(0, lo - 400), lo)
    above = range(hi + 1, min(UNICODE_MAX, hi + 400) + 1)
    for cp in list(below) + list(above):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        enc = chr(cp).encode("utf-8")
        assert not _matches(seqs, enc), f"U+{cp:04X} leaked into ({lo:#x},{hi:#x})"


@pytest.mark.parametrize(
    "lo,hi",
    [
        (0x00, 0x7F),
        (0x41, 0x5A),
        (0x00, 0x80),
        (0x80, 0x7FF),
        (0x80, 0x800),
        (0x100, 0x17F),
        (0x400, 0x4FF),
        (0x800, 0xFFFF),
        (0x3040, 0x30FF),
        (0xFF00, 0xFFEF),
        (0x10000, 0x1FFFF),
        (0x1F600, 0x1F64F),
        (0x0, 0x10FFFF),
        (0xD700, 0xE100),
        (0x2000, 0x206F),
    ],
)
def test_range_exact(lo: int, hi: int) -> None:
    _brute_check(lo, hi)


def test_full_range_sequence_count_is_small() -> None:
    # ``.`` should not blow up the automaton.
    assert len(utf8_sequences(0, UNICODE_MAX)) <= 20
