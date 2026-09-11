// bpdecode -- constraint-automaton token masking (host + device).
//
// This header defines the ABI: a scalar CPU implementation used as a
// reference, plus the batched CUDA kernels behind BPDECODE_WITH_CUDA.
#pragma once

#include <cstdint>
#include <vector>

namespace bpdecode {

// A deterministic finite automaton over "symbol classes" (disjoint code-point
// ranges), stored as a dense transition table in row-major order:
//   next_state = trans[state * num_symbols + symbol]
// `dead` is an absorbing non-accepting state. `live[s]` is true iff an
// accepting state is reachable from `s` (precomputed reachability -- the
// boolean message-passing fixpoint carried over from the BP kernels).
struct FsaTable {
  int32_t num_states = 0;
  int32_t num_symbols = 0;
  int32_t start = 0;
  int32_t dead = 0;
  std::vector<int32_t> trans;   // num_states * num_symbols
  std::vector<uint8_t> accept;  // num_states
  std::vector<uint8_t> live;    // num_states
};

// Token id -> the sequence of symbol-class ids its bytes drive through the FSA.
// Ragged: token `t` occupies [offsets[t], offsets[t+1]).
struct TokenSymbols {
  int32_t vocab_size = 0;
  int32_t eos_id = -1;
  std::vector<int32_t> offsets;  // vocab_size + 1
  std::vector<int32_t> symbols;  // offsets.back()
};

// Backward-reachability solve: fill `fsa.live` so that live[s] == 1 iff some
// accepting state is reachable from `s`. Boolean message passing to a fixpoint
// (live[s] |= OR over successors) -- the same pass the BP kernels ran, and the
// scalar oracle for `build_reachability_cuda`. Resizes `fsa.live` to num_states.
void build_reachability(FsaTable& fsa);

// Advance one DFA state by one token. Returns the next state, or -1 if the
// token is not accepted from `state`.
int32_t step(const FsaTable& fsa, const TokenSymbols& toks, int32_t state,
             int32_t token_id);

// Write the allow-mask for `state` into `out_bits` (bitset, LSB-first;
// out_bits must hold at least ceil(vocab_size / 32) uint32 words).
void compute_mask(const FsaTable& fsa, const TokenSymbols& toks, int32_t state,
                  uint32_t* out_bits);

// Batched form: one state per request, one mask row per request.
// out_bits is [batch * words_per_row] with words_per_row = ceil(vocab/32).
void compute_mask_batch(const FsaTable& fsa, const TokenSymbols& toks,
                        const int32_t* states, int32_t batch,
                        uint32_t* out_bits);

// Advance one state per request by the token that was actually sampled.
// next_states[b] = step(fsa, toks, states[b], token_ids[b]); a value of -1 means
// the sampled token was not on a grammar-valid path (mask bypassed) -- callers
// should treat that request as broken. EOS leaves the state unchanged.
void advance_state_batch(const FsaTable& fsa, const TokenSymbols& toks,
                         const int32_t* states, const int32_t* token_ids,
                         int32_t batch, int32_t* next_states);

// Fused mask + logit bias: logits is row-major [batch, vocab_size]; for each
// request every token not allowed from states[b] is set to `neg_inf`. Skips the
// bitset round-trip. `neg_inf` is caller-chosen (-INFINITY, or a large finite
// negative for fp16-safe softmax).
void apply_mask_batch(const FsaTable& fsa, const TokenSymbols& toks,
                      const int32_t* states, int32_t batch, float* logits,
                      float neg_inf);

#ifdef BPDECODE_WITH_CUDA
// Device path. Same semantics as the host functions above, but every array is
// a raw device pointer -- the SoA / CSR form the caller uploads once and keeps
// (there is no std::vector on the device). `stream` is a cudaStream_t.
//
//   trans    [num_states * num_symbols]  int32
//   accept   [num_states]                uint8
//   live     [num_states]                uint8   (written by build_reachability)
//   offsets  [vocab_size + 1]            int32
//   symbols  [offsets[vocab_size]]       int32

// live[s] |= OR(live[succ]) to a fixpoint.
void build_reachability_cuda(int32_t num_states, int32_t num_symbols,
                             const int32_t* trans, const uint8_t* accept,
                             uint8_t* live, void* stream);

void compute_mask_batch_cuda(int32_t num_states, int32_t num_symbols,
                             const int32_t* trans, const uint8_t* accept,
                             const uint8_t* live, int32_t dead,
                             const int32_t* offsets, const int32_t* symbols,
                             int32_t vocab_size, int32_t eos_id,
                             const int32_t* states, int32_t batch,
                             uint32_t* out_bits, void* stream);

void advance_state_batch_cuda(int32_t num_states, int32_t num_symbols,
                              const int32_t* trans, const uint8_t* accept,
                              const uint8_t* live, int32_t dead,
                              const int32_t* offsets, const int32_t* symbols,
                              int32_t vocab_size, int32_t eos_id,
                              const int32_t* states, const int32_t* token_ids,
                              int32_t batch, int32_t* next_states, void* stream);

void apply_mask_batch_cuda(int32_t num_states, int32_t num_symbols,
                           const int32_t* trans, const uint8_t* accept,
                           const uint8_t* live, int32_t dead,
                           const int32_t* offsets, const int32_t* symbols,
                           int32_t vocab_size, int32_t eos_id,
                           const int32_t* states, int32_t batch, float* logits,
                           float neg_inf, void* stream);
#endif

}  // namespace bpdecode
