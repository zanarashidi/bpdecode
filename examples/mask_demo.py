"""Step through a regex constraint by hand -- no model required.

    python examples/mask_demo.py
"""

from __future__ import annotations

from bpdecode import RegexConstraint, Vocabulary

# A toy byte-pair-ish vocab.
TOKENS = ["{", "}", '"', "name", "age", ":", ",", " ", "0", "1", "2", "<eos>"]
VOCAB = Vocabulary.from_tokens(TOKENS, eos_id=len(TOKENS) - 1)

# "an object with exactly one integer field called age"
PATTERN = r'\{"age":[0-9]+\}'


def show(con: RegexConstraint) -> None:
    allowed = sorted(TOKENS[i] for i in con.allowed_ids())
    flag = "  (complete)" if con.is_complete() else ""
    print(f"  allowed next: {allowed}{flag}")


def main() -> None:
    con = RegexConstraint(PATTERN, VOCAB)
    emitted: list[str] = []
    print(f"pattern: {PATTERN}\n")
    show(con)
    for tok in ["{", '"', "age", '"', ":", "1", "2", "}"]:
        tid = TOKENS.index(tok)
        assert con.accepts(tid), f"{tok!r} unexpectedly rejected"
        con.advance(tid)
        emitted.append(tok)
        print(f"emit {tok!r}  ->  {''.join(emitted)!r}")
        show(con)


if __name__ == "__main__":
    main()
