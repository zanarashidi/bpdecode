// Pushdown-automaton (CFG) kernels.
//
// Every device-side primitive (pda_close / pda_advance_byte / pda_try_token /
// ...) lives in pda.hpp as __host__ __device__ inline functions and is used
// here verbatim -- the kernels are thin per-thread wrappers around the exact
// same code csrc/src/pda_cpu.cpp calls, so CPU and CUDA cannot drift apart.
//
// One warp per request for the mask (lanes stride the vocab, each lane runs
// pda_try_token on its own stack-local PdaConfigSet copy -- config-sets are
// small and fixed-size, so this is register/local-memory bound, not shared
// memory). One thread per request for advance_state. Like the FSA kernels,
// these launch on the caller's stream and do not sync.
#include "bpdecode/pda.hpp"

#include <cstdio>

namespace bpdecode {

// A pointer-only view with the same member names pda.hpp's template functions
// index (`g.accept[s]`, `g.edge_offsets[s]`, `g.root_start`, ...), so
// pda_close & friends run unmodified against either PdaTable (host, owns
// vectors) or this (device, raw pointers into an uploaded SoA buffer).
struct PdaTableView {
  int32_t root_start;
  int32_t root_accept;
  const uint8_t* accept;
  const uint8_t* live;
  const int32_t* edge_offsets;
  const int32_t* edge_kind;
  const int32_t* edge_lo;
  const int32_t* edge_hi;
  const int32_t* edge_dst;
  const int32_t* edge_callee;
};

namespace {

constexpr int kWarp = 32;

__global__ void compute_mask_pda_kernel(
    PdaTableView g, int32_t vocab_size, int32_t eos_id,
    const int32_t* tok_offsets, const uint8_t* tok_bytes,
    const PdaConfigSet* configs, uint32_t* out_bits, int32_t words_per_row) {
  const int32_t req = blockIdx.x;
  const int32_t lane = threadIdx.x;  // blockDim.x == 32
  const PdaConfigSet base = configs[req];
  uint32_t* row = out_bits + static_cast<int64_t>(req) * words_per_row;

  for (int32_t word = 0; word * kWarp < vocab_size; ++word) {
    const int32_t tok = word * kWarp + lane;
    bool ok = false;
    if (tok < vocab_size) {
      if (tok == eos_id) {
        ok = pda_is_complete(g, base);
      } else {
        PdaConfigSet cfg = base;
        const int32_t begin = tok_offsets[tok];
        const int32_t end = tok_offsets[tok + 1];
        ok = true;
        for (int32_t i = begin; i < end && ok; ++i) {
          ok = pda_advance_byte(g, cfg, tok_bytes[i]);
        }
      }
    }
    const uint32_t packed = __ballot_sync(0xFFFFFFFFu, ok);
    if (lane == 0) row[word] = packed;
  }
}

// Fused mask + logit bias, no bitset: lanes stride the vocab, each writes
// neg_inf for its own disallowed tokens directly into logits.
__global__ void apply_mask_pda_kernel(
    PdaTableView g, int32_t vocab_size, int32_t eos_id,
    const int32_t* tok_offsets, const uint8_t* tok_bytes,
    const PdaConfigSet* configs, float* logits, float neg_inf) {
  const int32_t req = blockIdx.x;
  const PdaConfigSet base = configs[req];
  float* row = logits + static_cast<int64_t>(req) * vocab_size;

  for (int32_t tok = threadIdx.x; tok < vocab_size; tok += blockDim.x) {
    bool ok;
    if (tok == eos_id) {
      ok = pda_is_complete(g, base);
    } else {
      PdaConfigSet cfg = base;
      const int32_t begin = tok_offsets[tok];
      const int32_t end = tok_offsets[tok + 1];
      ok = true;
      for (int32_t i = begin; i < end && ok; ++i) {
        ok = pda_advance_byte(g, cfg, tok_bytes[i]);
      }
    }
    if (!ok) row[tok] = neg_inf;
  }
}

__global__ void advance_state_pda_kernel(
    PdaTableView g, int32_t eos_id, const int32_t* tok_offsets,
    const uint8_t* tok_bytes, PdaConfigSet* configs, const int32_t* token_ids,
    int32_t batch, uint8_t* ok) {
  const int32_t b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch) return;
  const int32_t tok = token_ids[b];
  if (tok == eos_id) {
    ok[b] = pda_is_complete(g, configs[b]) ? 1 : 0;
    return;
  }
  PdaConfigSet trial = configs[b];
  bool accepted = true;
  const int32_t begin = tok_offsets[tok];
  const int32_t end = tok_offsets[tok + 1];
  for (int32_t i = begin; i < end && accepted; ++i) {
    accepted = pda_advance_byte(g, trial, tok_bytes[i]);
  }
  ok[b] = accepted ? 1 : 0;
  if (accepted) configs[b] = trial;
}

inline void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "bpdecode CUDA (pda): %s: %s\n", what, cudaGetErrorString(e));
  }
}

}  // namespace

void compute_mask_pda_batch_cuda(
    int32_t num_states, int32_t root_start, int32_t root_accept,
    const uint8_t* accept, const uint8_t* live, const int32_t* edge_offsets,
    const int32_t* edge_kind, const int32_t* edge_lo, const int32_t* edge_hi,
    const int32_t* edge_dst, const int32_t* edge_callee, int32_t vocab_size,
    int32_t eos_id, const int32_t* tok_offsets, const uint8_t* tok_bytes,
    const PdaConfigSet* configs, int32_t batch, uint32_t* out_bits, void* stream) {
  if (batch <= 0) return;
  (void)num_states;
  auto s = static_cast<cudaStream_t>(stream);
  const int32_t words_per_row = (vocab_size + 31) / 32;
  PdaTableView g{root_start,  root_accept, accept,    live,
                edge_offsets, edge_kind,   edge_lo,   edge_hi,
                edge_dst,     edge_callee};
  check(cudaMemsetAsync(out_bits, 0,
                        static_cast<size_t>(batch) * words_per_row *
                            sizeof(uint32_t),
                        s),
        "memset pda mask");
  compute_mask_pda_kernel<<<batch, kWarp, 0, s>>>(
      g, vocab_size, eos_id, tok_offsets, tok_bytes, configs, out_bits,
      words_per_row);
}

void apply_mask_pda_batch_cuda(
    int32_t num_states, int32_t root_start, int32_t root_accept,
    const uint8_t* accept, const uint8_t* live, const int32_t* edge_offsets,
    const int32_t* edge_kind, const int32_t* edge_lo, const int32_t* edge_hi,
    const int32_t* edge_dst, const int32_t* edge_callee, int32_t vocab_size,
    int32_t eos_id, const int32_t* tok_offsets, const uint8_t* tok_bytes,
    const PdaConfigSet* configs, int32_t batch, float* logits, float neg_inf,
    void* stream) {
  if (batch <= 0) return;
  (void)num_states;
  auto s = static_cast<cudaStream_t>(stream);
  PdaTableView g{root_start,  root_accept, accept,    live,
                edge_offsets, edge_kind,   edge_lo,   edge_hi,
                edge_dst,     edge_callee};
  apply_mask_pda_kernel<<<batch, kWarp, 0, s>>>(
      g, vocab_size, eos_id, tok_offsets, tok_bytes, configs, logits, neg_inf);
}

void advance_state_pda_batch_cuda(
    int32_t num_states, int32_t root_start, int32_t root_accept,
    const uint8_t* accept, const uint8_t* live, const int32_t* edge_offsets,
    const int32_t* edge_kind, const int32_t* edge_lo, const int32_t* edge_hi,
    const int32_t* edge_dst, const int32_t* edge_callee, int32_t vocab_size,
    int32_t eos_id, const int32_t* tok_offsets, const uint8_t* tok_bytes,
    PdaConfigSet* configs, const int32_t* token_ids, int32_t batch, uint8_t* ok,
    void* stream) {
  if (batch <= 0) return;
  (void)num_states;
  (void)vocab_size;
  auto s = static_cast<cudaStream_t>(stream);
  PdaTableView g{root_start,  root_accept, accept,    live,
                edge_offsets, edge_kind,   edge_lo,   edge_hi,
                edge_dst,     edge_callee};
  const int32_t block = 128;
  const int32_t grid = (batch + block - 1) / block;
  advance_state_pda_kernel<<<grid, block, 0, s>>>(
      g, eos_id, tok_offsets, tok_bytes, configs, token_ids, batch, ok);
}

}  // namespace bpdecode
