#include "bpdecode/mask.hpp"

#include <cstring>
#include <vector>

namespace bpdecode {

void build_reachability(FsaTable& fsa) {
  const int32_t n = fsa.num_states;
  const int32_t m = fsa.num_symbols;
  fsa.live.assign(static_cast<size_t>(n), 0);

  // Reverse adjacency (CSR) so each sweep is O(edges), not O(states*symbols).
  std::vector<int32_t> pred_count(static_cast<size_t>(n) + 1, 0);
  for (int32_t s = 0; s < n; ++s) {
    for (int32_t k = 0; k < m; ++k) {
      const int32_t t = fsa.trans[static_cast<int64_t>(s) * m + k];
      if (t >= 0 && t < n) ++pred_count[t + 1];
    }
  }
  for (int32_t i = 0; i < n; ++i) pred_count[i + 1] += pred_count[i];
  std::vector<int32_t> preds(pred_count[n]);
  std::vector<int32_t> cursor(pred_count.begin(), pred_count.end() - 1);
  for (int32_t s = 0; s < n; ++s) {
    for (int32_t k = 0; k < m; ++k) {
      const int32_t t = fsa.trans[static_cast<int64_t>(s) * m + k];
      if (t >= 0 && t < n) preds[cursor[t]++] = s;
    }
  }

  std::vector<int32_t> frontier;
  for (int32_t s = 0; s < n; ++s) {
    if (fsa.accept[s]) {
      fsa.live[s] = 1;
      frontier.push_back(s);
    }
  }
  while (!frontier.empty()) {
    const int32_t cur = frontier.back();
    frontier.pop_back();
    for (int32_t i = pred_count[cur]; i < pred_count[cur + 1]; ++i) {
      const int32_t p = preds[i];
      if (!fsa.live[p]) {
        fsa.live[p] = 1;
        frontier.push_back(p);
      }
    }
  }
}

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
