"""Model-probability-weighted soft lookahead, as a real batched generation
loop instead of the single-request, no-cache proof of concept in
``bench/soft_eval_modelweighted.py`` (see that file / ``bench/RESULTS.md``
for why the count-based ``ops.build_lookahead`` bias doesn't work, and why
weighting by the model's own probability does).

Hard masking is a pure function of grammar state; nothing here can improve
its *validity*. What it can do is stop the model from committing to a
grammar-valid token that only has grammar-dead continuations further out
(the same trap byte-level reachability can fall into), by scoring each
top-K allowed candidate with one extra forward pass over *its* next-step
distribution:

    score(t) = logit(t) + alpha * log P(next token is grammar-valid | took t)

Why this needs its own generation loop rather than a ``LogitsProcessor``:
``model.generate()`` does not hand its KV cache to logits processors, so a
processor-based version would have no way to reuse it and would recompute
the whole prefix for every candidate, every step -- fine for a short demo,
unusable for real generation length. Here the cache is forked once per step
(``DynamicCache.batch_repeat_interleave``), the K candidate branches share
one batched forward call, and the winning branch's cache **is** the next
step's cache (``batch_select_indices`` prunes the K-1 losers) -- so the
added cost is exactly K forward passes per decode step, not K prefix
recomputations.

Regex grammars only (works directly against ``FsaTensors`` state, which is
a plain int tensor -- cheap to repeat/select for branching). CFG grammars
would need the same treatment against ``PdaTensors`` config-sets; not done
here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .ops import FsaTensors, advance_state, apply_mask_
from .regex.compile import DFA
from .tokenizer import Vocabulary


def _score_candidates(
    cand_logits: torch.Tensor, weight: torch.Tensor, alpha: float
) -> torch.Tensor:
    """``cand_logits + alpha * weight``, except a candidate with ``weight ==
    -inf`` (no valid continuation at all -- a dead end, or itself an
    already-disallowed candidate whose branch state came back BROKEN) always
    scores ``-inf``, regardless of ``alpha``. Plain ``cand_logits + alpha *
    weight`` would give ``nan`` at ``alpha == 0`` (``0 * -inf``), not just
    drop the dead-end penalty -- pulled out as its own function so this is
    unit-testable without a model.
    """
    dead = torch.isneginf(weight)
    contrib = weight.masked_fill(dead, 0.0) * alpha
    return (cand_logits + contrib).masked_fill(dead, float("-inf"))


@torch.no_grad()
def generate_model_weighted(
    model: object,
    tokenizer: object,
    prompts: list[str],
    pattern: str | DFA,
    *,
    vocab: Vocabulary | None = None,
    k: int = 6,
    alpha: float = 1.0,
    max_new_tokens: int = 64,
    device: torch.device | str | None = None,
) -> list[str]:
    """Batch-generate ``len(prompts)`` completions constrained to ``pattern``,
    each step choosing among the top ``k`` grammar-allowed tokens by
    ``alpha``-weighted model probability of staying grammar-valid one more
    step. A candidate with *no* valid continuation at all (a token-level
    dead end hard masking alone can't see) is excluded regardless of
    ``alpha``; ``alpha=0`` drops only the soft magnitude preference among
    the still-viable candidates, so it is hard-masked greedy decoding plus
    that one-step dead-end check, not pure hard masking.

    Runs a fixed ``max_new_tokens`` steps for every row (no ragged early
    stop across the batch -- simplest correct batching); text after the
    first EOS in each row is truncated on return.
    """
    if alpha < 0:
        raise ValueError("alpha must be >= 0 (this mode only biases toward validity)")
    from transformers.cache_utils import DynamicCache

    dev = torch.device(device) if device is not None else next(model.parameters()).device
    vocab = vocab or Vocabulary.from_hf(tokenizer)
    fsa = FsaTensors.build(pattern, vocab).densify().to(dev)
    v = fsa.vocab_size
    n = len(prompts)

    orig_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        enc = tokenizer(prompts, return_tensors="pt", padding=True)
    finally:
        tokenizer.padding_side = orig_side
    input_ids = enc["input_ids"].to(dev)
    attn_mask = enc["attention_mask"].to(dev)
    position_ids = (attn_mask.long().cumsum(-1) - 1).clamp_(min=0)

    cache = DynamicCache()
    out = model(
        input_ids, attention_mask=attn_mask, position_ids=position_ids,
        past_key_values=cache, use_cache=True,
    )
    logits = out.logits[:, -1, :v].clone().float()
    cache = out.past_key_values

    states = torch.full((n,), fsa.start, dtype=torch.int32, device=dev)
    generated = torch.zeros(n, max_new_tokens, dtype=torch.long, device=dev)

    for step in range(max_new_tokens):
        apply_mask_(logits, fsa, states)  # in place; -inf where disallowed

        kk = min(k, v)
        probe = torch.where(torch.isneginf(logits), torch.finfo(logits.dtype).min, logits)
        cand_ids = torch.topk(probe, kk, dim=1).indices  # [n, kk]
        cand_logits = logits.gather(1, cand_ids)  # [n, kk], -inf where disallowed

        cur_len = attn_mask.shape[1]
        cache.batch_repeat_interleave(kk)
        branch_states = advance_state(
            fsa, states.repeat_interleave(kk), cand_ids.reshape(-1)
        )
        pad_col = torch.ones(n * kk, 1, dtype=attn_mask.dtype, device=dev)
        branch_attn = torch.cat(
            [attn_mask.repeat_interleave(kk, dim=0), pad_col], dim=1
        )
        branch_pos = torch.full((n * kk, 1), cur_len, dtype=torch.long, device=dev)
        cache_position = torch.tensor([cur_len], dtype=torch.long, device=dev)

        branch_out = model(
            cand_ids.reshape(-1, 1), attention_mask=branch_attn,
            position_ids=branch_pos, past_key_values=cache, use_cache=True,
            cache_position=cache_position,
        )
        next_logits = branch_out.logits[:, -1, :v].clone().float()
        cache = branch_out.past_key_values
        apply_mask_(next_logits, fsa, branch_states)
        weight = torch.where(
            torch.isneginf(next_logits), torch.full_like(next_logits, float("-inf")),
            F.log_softmax(next_logits, dim=-1),
        ).logsumexp(dim=-1)  # [n*kk], log P(next valid | took candidate)

        score = _score_candidates(cand_logits.reshape(-1), weight, alpha).view(n, kk)
        best = score.argmax(dim=1)  # [n]
        chosen = cand_ids.gather(1, best.unsqueeze(1)).squeeze(1)  # [n]

        winner_idx = torch.arange(n, device=dev) * kk + best
        cache.batch_select_indices(winner_idx)
        logits = next_logits.view(n, kk, v)[torch.arange(n, device=dev), best]
        states = branch_states.view(n, kk)[torch.arange(n, device=dev), best]
        attn_mask = torch.cat(
            [attn_mask, torch.ones(n, 1, dtype=attn_mask.dtype, device=dev)], dim=1
        )
        generated[:, step] = chosen

    texts = []
    eos = vocab.eos_id
    for row in generated.tolist():
        if eos is not None and eos in row:
            row = row[: row.index(eos)]
        texts.append(tokenizer.decode(row, skip_special_tokens=True))
    return texts
