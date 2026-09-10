// bpdecode -- constraint-automaton token masking (host + device).
//
// Phase 0 defines the ABI and ships a scalar CPU implementation used as a
// reference. The batched CUDA kernels land in Phase 1 behind BPDECODE_WITH_CUDA.
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

#ifdef BPDECODE_WITH_CUDA
// Phase 1: same contract, device pointers. Declared now so callers can compile
// against the final ABI.
void compute_mask_batch_cuda(const FsaTable& fsa_device,
                             const TokenSymbols& toks_device,
                             const int32_t* states_device, int32_t batch,
                             uint32_t* out_bits_device, void* stream);

// Iterative boolean-BP reachability on device. `trans`/`accept` are device
// pointers of length num_states*num_symbols / num_states; `live_device` (length
// num_states) is written. Runs live[s] |= OR(live[succ]) to a fixpoint.
void build_reachability_cuda(int32_t num_states, int32_t num_symbols,
                             const int32_t* trans_device,
                             const uint8_t* accept_device,
                             uint8_t* live_device, void* stream);
#endif

}  // namespace bpdecode
