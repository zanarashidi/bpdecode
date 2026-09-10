// Batched constraint-mask kernels.
//
// Layout / modernization choices (see docs/PLAN.md):
//   * one warp per request, lanes stride over the vocab
//   * __ballot_sync packs 32 token verdicts into one uint32 mask word
//   * transition table read straight from global memory as SoA / CSR
//   * caller-supplied stream so the mask compute overlaps the model forward pass
//
// Every array argument is a raw device pointer -- the SoA form the caller
// uploads once and keeps. See the CUDA section of mask.hpp for the layouts.
//
// The per-step entry points (compute_mask / advance_state / apply_mask) launch
// on `stream` and return without synchronising -- so the mask can overlap the
// model's forward pass on another stream. Ordering vs. later work on the same
// stream is guaranteed by CUDA; the caller syncs before it reads the result
// (e.g. before sampling). build_reachability is a one-time compile step and
// does sync (it reads a device flag to decide when its fixpoint has converged).
#include "bpdecode/mask.hpp"

#include <cstdio>

namespace bpdecode {

namespace {

constexpr int kWarp = 32;

__device__ inline int32_t step_device(const int32_t* trans, int32_t num_symbols,
                                      int32_t num_states, const uint8_t* live,
                                      int32_t dead, const int32_t* offsets,
                                      const int32_t* symbols, int32_t eos_id,
                                      const uint8_t* accept, int32_t state,
                                      int32_t token_id) {
  if (token_id == eos_id) {
    return (state >= 0 && accept[state]) ? state : -1;
  }
  if (state < 0) return -1;
  int32_t cur = state;
  const int32_t begin = offsets[token_id];
  const int32_t end = offsets[token_id + 1];
  for (int32_t i = begin; i < end; ++i) {
    const int32_t sym = symbols[i];
    if (sym < 0 || sym >= num_symbols) return -1;
    cur = trans[static_cast<int64_t>(cur) * num_symbols + sym];
    if (cur == dead) return -1;
  }
  (void)num_states;
  return live[cur] ? cur : -1;
}

// grid.x == batch, one warp per block. Each lane owns tokens {lane, lane+32,...}
// and contributes its bit into the packed word via __ballot_sync.
__global__ void compute_mask_kernel(const int32_t* trans, int32_t num_symbols,
                                    int32_t num_states, const uint8_t* accept,
                                    const uint8_t* live, int32_t dead,
                                    const int32_t* offsets,
                                    const int32_t* symbols, int32_t vocab_size,
                                    int32_t eos_id, const int32_t* states,
                                    uint32_t* out_bits, int32_t words_per_row) {
  const int32_t req = blockIdx.x;
  const int32_t lane = threadIdx.x;  // blockDim.x == 32
  const int32_t state = states[req];
  uint32_t* row = out_bits + static_cast<int64_t>(req) * words_per_row;

  for (int32_t word = 0; word * kWarp < vocab_size; ++word) {
    const int32_t tok = word * kWarp + lane;
    bool ok = false;
    if (tok < vocab_size) {
      ok = step_device(trans, num_symbols, num_states, live, dead, offsets,
                       symbols, eos_id, accept, state, tok) != -1;
    }
    const uint32_t packed = __ballot_sync(0xFFFFFFFFu, ok);
    if (lane == 0) row[word] = packed;
  }

  // EOS may sit anywhere in the vocab; make sure its bit reflects `accept`.
  if (lane == 0 && eos_id >= 0 && state >= 0 && accept[state]) {
    row[eos_id >> 5] |= (1u << (eos_id & 31));
  }
}

// One thread per state. Repeatedly OR in successors' liveness until no thread
// flips a bit this round (fixpoint) -- the boolean BP loop, CUDA-graph friendly.
__global__ void reachability_step_kernel(int32_t num_states, int32_t num_symbols,
                                         const int32_t* trans,
                                         const uint8_t* accept, uint8_t* live,
                                         int32_t* changed) {
  const int32_t s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= num_states) return;
  if (live[s]) return;
  uint8_t v = accept[s];
  if (!v) {
    const int64_t base = static_cast<int64_t>(s) * num_symbols;
    for (int32_t k = 0; k < num_symbols && !v; ++k) {
      const int32_t t = trans[base + k];
      if (t >= 0 && t < num_states && live[t]) v = 1;
    }
  }
  if (v) {
    live[s] = 1;
    *changed = 1;
  }
}

// One thread per request: commit the sampled token, write the next state.
__global__ void advance_state_kernel(const int32_t* trans, int32_t num_symbols,
                                     int32_t num_states, const uint8_t* accept,
                                     const uint8_t* live, int32_t dead,
                                     const int32_t* offsets,
                                     const int32_t* symbols, int32_t eos_id,
                                     const int32_t* states,
                                     const int32_t* token_ids, int32_t batch,
                                     int32_t* next_states) {
  const int32_t b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch) return;
  next_states[b] = step_device(trans, num_symbols, num_states, live, dead,
                               offsets, symbols, eos_id, accept, states[b],
                               token_ids[b]);
}

// One warp per request, lanes stride the vocab: push every disallowed logit to
// neg_inf in place. Fused -- no bitset materialised.
__global__ void apply_mask_kernel(const int32_t* trans, int32_t num_symbols,
                                  int32_t num_states, const uint8_t* accept,
                                  const uint8_t* live, int32_t dead,
                                  const int32_t* offsets, const int32_t* symbols,
                                  int32_t vocab_size, int32_t eos_id,
                                  const int32_t* states, float* logits,
                                  float neg_inf) {
  const int32_t req = blockIdx.x;
  const int32_t state = states[req];
  float* row = logits + static_cast<int64_t>(req) * vocab_size;
  for (int32_t tok = threadIdx.x; tok < vocab_size; tok += blockDim.x) {
    if (step_device(trans, num_symbols, num_states, live, dead, offsets, symbols,
                    eos_id, accept, state, tok) == -1) {
      row[tok] = neg_inf;
    }
  }
}

inline void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "bpdecode CUDA: %s: %s\n", what, cudaGetErrorString(e));
  }
}

}  // namespace

void build_reachability_cuda(int32_t num_states, int32_t num_symbols,
                             const int32_t* trans, const uint8_t* accept,
                             uint8_t* live, void* stream) {
  auto s = static_cast<cudaStream_t>(stream);
  check(cudaMemsetAsync(live, 0, static_cast<size_t>(num_states), s),
        "memset live");

  int32_t* changed = nullptr;
  check(cudaMallocAsync(&changed, sizeof(int32_t), s), "alloc flag");

  const int32_t block = 256;
  const int32_t grid = (num_states + block - 1) / block;
  int32_t host_changed = 1;
  // Bounded by the automaton diameter; num_states is a safe hard cap.
  for (int32_t iter = 0; iter < num_states && host_changed; ++iter) {
    check(cudaMemsetAsync(changed, 0, sizeof(int32_t), s), "reset flag");
    reachability_step_kernel<<<grid, block, 0, s>>>(num_states, num_symbols,
                                                    trans, accept, live, changed);
    check(cudaMemcpyAsync(&host_changed, changed, sizeof(int32_t),
                          cudaMemcpyDeviceToHost, s),
          "copy flag");
    check(cudaStreamSynchronize(s), "sync");
  }
  check(cudaFreeAsync(changed, s), "free flag");
}

void compute_mask_batch_cuda(int32_t num_states, int32_t num_symbols,
                             const int32_t* trans, const uint8_t* accept,
                             const uint8_t* live, int32_t dead,
                             const int32_t* offsets, const int32_t* symbols,
                             int32_t vocab_size, int32_t eos_id,
                             const int32_t* states, int32_t batch,
                             uint32_t* out_bits, void* stream) {
  if (batch <= 0) return;
  const int32_t words_per_row = (vocab_size + 31) / 32;
  auto s = static_cast<cudaStream_t>(stream);

  check(cudaMemsetAsync(out_bits, 0,
                        static_cast<size_t>(batch) * words_per_row *
                            sizeof(uint32_t),
                        s),
        "memset mask");
  compute_mask_kernel<<<batch, kWarp, 0, s>>>(
      trans, num_symbols, num_states, accept, live, dead, offsets, symbols,
      vocab_size, eos_id, states, out_bits, words_per_row);
}

void advance_state_batch_cuda(int32_t num_states, int32_t num_symbols,
                              const int32_t* trans, const uint8_t* accept,
                              const uint8_t* live, int32_t dead,
                              const int32_t* offsets, const int32_t* symbols,
                              int32_t vocab_size, int32_t eos_id,
                              const int32_t* states, const int32_t* token_ids,
                              int32_t batch, int32_t* next_states, void* stream) {
  if (batch <= 0) return;
  (void)vocab_size;
  auto s = static_cast<cudaStream_t>(stream);
  const int32_t block = 128;
  const int32_t grid = (batch + block - 1) / block;
  advance_state_kernel<<<grid, block, 0, s>>>(
      trans, num_symbols, num_states, accept, live, dead, offsets, symbols,
      eos_id, states, token_ids, batch, next_states);
}

void apply_mask_batch_cuda(int32_t num_states, int32_t num_symbols,
                           const int32_t* trans, const uint8_t* accept,
                           const uint8_t* live, int32_t dead,
                           const int32_t* offsets, const int32_t* symbols,
                           int32_t vocab_size, int32_t eos_id,
                           const int32_t* states, int32_t batch, float* logits,
                           float neg_inf, void* stream) {
  if (batch <= 0) return;
  auto s = static_cast<cudaStream_t>(stream);
  apply_mask_kernel<<<batch, kWarp, 0, s>>>(
      trans, num_symbols, num_states, accept, live, dead, offsets, symbols,
      vocab_size, eos_id, states, logits, neg_inf);
}

}  // namespace bpdecode
