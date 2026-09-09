#include "bpdecode/mask.hpp"

#include <cstring>

namespace bpdecode {

int32_t step(const FsaTable& fsa, const TokenSymbols& toks, int32_t state,
             int32_t token_id) {
  if (token_id == toks.eos_id) {
    return (state >= 0 && fsa.accept[state]) ? state : -1;
  }
  if (state < 0) return -1;
  int32_t cur = state;
  const int32_t begin = toks.offsets[token_id];
  const int32_t end = toks.offsets[token_id + 1];
  for (int32_t i = begin; i < end; ++i) {
    const int32_t sym = toks.symbols[i];
    if (sym < 0 || sym >= fsa.num_symbols) return -1;
    cur = fsa.trans[static_cast<int64_t>(cur) * fsa.num_symbols + sym];
    if (cur == fsa.dead) return -1;
  }
  return fsa.live[cur] ? cur : -1;
}

void compute_mask(const FsaTable& fsa, const TokenSymbols& toks, int32_t state,
                  uint32_t* out_bits) {
  const int32_t words = (toks.vocab_size + 31) / 32;
  std::memset(out_bits, 0, static_cast<size_t>(words) * sizeof(uint32_t));
  for (int32_t t = 0; t < toks.vocab_size; ++t) {
    if (step(fsa, toks, state, t) != -1) {
      out_bits[t >> 5] |= (1u << (t & 31));
    }
  }
  if (toks.eos_id >= 0 && state >= 0 && fsa.accept[state]) {
    out_bits[toks.eos_id >> 5] |= (1u << (toks.eos_id & 31));
  }
}

void compute_mask_batch(const FsaTable& fsa, const TokenSymbols& toks,
                        const int32_t* states, int32_t batch,
                        uint32_t* out_bits) {
  const int32_t words = (toks.vocab_size + 31) / 32;
  for (int32_t b = 0; b < batch; ++b) {
    compute_mask(fsa, toks, states[b], out_bits + static_cast<int64_t>(b) * words);
  }
}

}  // namespace bpdecode
