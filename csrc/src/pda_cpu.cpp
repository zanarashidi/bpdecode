#include "bpdecode/pda.hpp"

#include <cstring>

namespace bpdecode {

void compute_mask_pda_batch(const PdaTable& g, const PdaTokens& toks,
                            const PdaConfigSet* configs, int32_t batch,
                            uint32_t* out_bits) {
  const int32_t words = (toks.vocab_size + 31) / 32;
  std::memset(out_bits, 0, static_cast<size_t>(batch) * words * sizeof(uint32_t));
  for (int32_t b = 0; b < batch; ++b) {
    uint32_t* row = out_bits + static_cast<int64_t>(b) * words;
    for (int32_t t = 0; t < toks.vocab_size; ++t) {
      if (pda_try_token(g, toks, configs[b], t)) row[t >> 5] |= (1u << (t & 31));
    }
    if (toks.eos_id >= 0 && pda_is_complete(g, configs[b])) {
      row[toks.eos_id >> 5] |= (1u << (toks.eos_id & 31));
    }
  }
}

void apply_mask_pda_batch(const PdaTable& g, const PdaTokens& toks,
                          const PdaConfigSet* configs, int32_t batch,
                          float* logits, float neg_inf) {
  const int64_t vocab = toks.vocab_size;
  for (int32_t b = 0; b < batch; ++b) {
    float* row = logits + b * vocab;
    for (int32_t t = 0; t < toks.vocab_size; ++t) {
      if (!pda_try_token(g, toks, configs[b], t)) row[t] = neg_inf;
    }
  }
}

void advance_state_pda_batch(const PdaTable& g, const PdaTokens& toks,
                             PdaConfigSet* configs, const int32_t* token_ids,
                             int32_t batch, uint8_t* ok) {
  for (int32_t b = 0; b < batch; ++b) {
    const int32_t t = token_ids[b];
    if (t == toks.eos_id) {
      ok[b] = pda_is_complete(g, configs[b]) ? 1 : 0;
      continue;  // EOS never mutates the config-set
    }
    PdaConfigSet trial = configs[b];
    bool accepted = true;
    const int32_t begin = toks.offsets[t];
    const int32_t end = toks.offsets[t + 1];
    for (int32_t i = begin; i < end && accepted; ++i) {
      accepted = pda_advance_byte(g, trial, toks.bytes[i]);
    }
    ok[b] = accepted ? 1 : 0;
    if (accepted) configs[b] = trial;
  }
}

}  // namespace bpdecode
