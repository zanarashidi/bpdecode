"""Lower a Unicode code-point range to an automaton over UTF-8 bytes.

Constrained decoding runs on raw token bytes, so a regex written over code
points has to become an automaton over bytes.  A single code-point range
expands to a small set of alternatives, each a sequence of inclusive byte
ranges -- e.g. ``U+0080..U+07FF`` is ``[C2-DF][80-BF]`` and ``U+0000..U+10FFFF``
(``.``) is a dozen such sequences covering every valid scalar.

This is the standard UTF-8 range split (cf. Rust's ``regex-syntax``
``Utf8Sequences``), done here by recursion on byte position so it can be
checked exhaustively against the platform codec.
"""

from __future__ import annotations

_MAX_1, _MAX_2, _MAX_3 = 0x7F, 0x7FF, 0xFFFF
_SURROGATE_LO, _SURROGATE_HI = 0xD800, 0xDFFF
UNICODE_MAX = 0x10FFFF

ByteRange = tuple[int, int]
ByteSeq = tuple[ByteRange, ...]


def _encode(cp: int) -> list[int]:
    if cp <= _MAX_1:
        return [cp]
    if cp <= _MAX_2:
        return [0xC0 | (cp >> 6), 0x80 | (cp & 0x3F)]
    if cp <= _MAX_3:
        return [0xE0 | (cp >> 12), 0x80 | ((cp >> 6) & 0x3F), 0x80 | (cp & 0x3F)]
    return [
        0xF0 | (cp >> 18),
        0x80 | ((cp >> 12) & 0x3F),
        0x80 | ((cp >> 6) & 0x3F),
        0x80 | (cp & 0x3F),
    ]


def _clip_surrogates(lo: int, hi: int) -> list[tuple[int, int]]:
    lo, hi = max(lo, 0), min(hi, UNICODE_MAX)
    if lo > hi:
        return []
    if hi < _SURROGATE_LO or lo > _SURROGATE_HI:
        return [(lo, hi)]
    out: list[tuple[int, int]] = []
    if lo < _SURROGATE_LO:
        out.append((lo, _SURROGATE_LO - 1))
    if hi > _SURROGATE_HI:
        out.append((_SURROGATE_HI + 1, hi))
    return out


def _by_length(lo: int, hi: int) -> list[tuple[int, int]]:
    """Split so both endpoints encode to the same number of bytes."""
    out: list[tuple[int, int]] = []
    start = lo
    for boundary in (_MAX_1, _MAX_2, _MAX_3):
        if start <= boundary < hi:
            out.append((start, boundary))
            start = boundary + 1
    out.append((start, hi))
    return out


def _gen(
    pos: int,
    n: int,
    lo_bytes: list[int],
    hi_bytes: list[int],
    tight_lo: bool,
    tight_hi: bool,
    prefix: ByteSeq,
    out: list[ByteSeq],
) -> None:
    if not tight_lo and not tight_hi:
        # every remaining byte spans the full continuation range
        out.append((*prefix, *(((0x80, 0xBF),) * (n - pos))))
        return

    lo_b = lo_bytes[pos] if tight_lo else 0x80
    hi_b = hi_bytes[pos] if tight_hi else 0xBF

    if pos == n - 1:
        out.append((*prefix, (lo_b, hi_b)))
        return

    if lo_b == hi_b:
        _gen(
            pos + 1, n, lo_bytes, hi_bytes,
            tight_lo, tight_hi,
            (*prefix, (lo_b, lo_b)), out,
        )
        return

    rest = n - 1 - pos
    tail = ((0x80, 0xBF),) * rest
    # the low / high leading byte can join the interior when its remaining
    # bytes already span the full continuation range
    lo_full = tight_lo and all(lo_bytes[k] == 0x80 for k in range(pos + 1, n))
    hi_full = tight_hi and all(hi_bytes[k] == 0xBF for k in range(pos + 1, n))

    if not lo_full:
        _gen(pos + 1, n, lo_bytes, hi_bytes, tight_lo, False,
             (*prefix, (lo_b, lo_b)), out)
    if not hi_full:
        _gen(pos + 1, n, lo_bytes, hi_bytes, False, tight_hi,
             (*prefix, (hi_b, hi_b)), out)

    inner_lo = lo_b if lo_full else lo_b + 1
    inner_hi = hi_b if hi_full else hi_b - 1
    if inner_lo <= inner_hi:
        out.append((*prefix, (inner_lo, inner_hi), *tail))


def utf8_sequences(lo: int, hi: int) -> list[ByteSeq]:
    """Byte-range sequences whose decodings are exactly the scalars in ``[lo, hi]``.

    Surrogates (``U+D800..U+DFFF``) are excluded -- they have no UTF-8 form.
    """
    out: list[ByteSeq] = []
    for a, b in _clip_surrogates(lo, hi):
        for x, y in _by_length(a, b):
            xb, yb = _encode(x), _encode(y)
            _gen(0, len(xb), xb, yb, True, True, (), out)
    return out
