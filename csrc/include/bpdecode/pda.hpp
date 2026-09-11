// bpdecode -- pushdown-automaton (CFG) token masking (host + device).
//
// The runtime state is a *bounded set of stacks* (see bpdecode.grammar.pda.PDA
// for the unbounded Python reference this mirrors): each stack is a sequence
// of global state ids, one per active rule frame. State ids are global across
// every rule's NFA -- rule R's states occupy a contiguous slice of
// [0, num_states) -- so a single flat transition table covers the whole
// grammar and a "call" edge just names another state's id as its push target.
//
// kPdaMaxConfigs / kPdaMaxDepth cap the config-set size and recursion depth so
// the whole state fits in fixed-size, device-friendly arrays (no STL in the
// hot path). Real grammars (JSON Schema, GBNF without deliberate ambiguity)
// stay far under these; a grammar that needs more silently saturates -- config
// expansions past the cap are dropped rather than raising, since device code
// can't throw. `pda_close` returning with `num_configs == kPdaMaxConfigs` is a
// signal something may have been dropped; check `PdaConfigSet::saturated`.
//
// The closure / step primitives below are written once, host+device, and used
// verbatim by both csrc/src/pda_cpu.cpp and csrc/src/pda_cuda.cu -- the CPU
// reference and the CUDA kernel cannot drift apart because they share the
// literal same code.
#pragma once

#include <cstdint>
#include <vector>

#if defined(__CUDACC__)
#define BPDECODE_HD __host__ __device__
#else
#define BPDECODE_HD
#endif

namespace bpdecode {

constexpr int32_t kPdaMaxConfigs = 8;
constexpr int32_t kPdaMaxDepth = 32;

enum PdaEdgeKind : int32_t { kPdaEps = 0, kPdaByte = 1, kPdaCall = 2 };

// Flattened grammar: every rule's NFA concatenated into one global state space
// plus a CSR edge list. `accept[s]` marks a state where its rule may pop
// (equivalent to RuleNFA.accept in grammar/nfa.py); `live[s]` is the
// coreachability prune from grammar/pda.py's CompiledGrammar (states from
// which the *rule* can still complete, calls counted passable iff the callee
// is non-empty) -- both computed once on the host.
struct PdaTable {
  int32_t num_states = 0;
  int32_t root_start = 0;
  int32_t root_accept = 0;
  std::vector<uint8_t> accept;        // [num_states]
  std::vector<uint8_t> live;          // [num_states]
  std::vector<int32_t> edge_offsets;  // [num_states + 1]
  std::vector<int32_t> edge_kind;     // [nnz]  PdaEdgeKind
  std::vector<int32_t> edge_lo;       // [nnz]  byte range lo (kPdaByte only)
  std::vector<int32_t> edge_hi;       // [nnz]  byte range hi (kPdaByte only)
  std::vector<int32_t> edge_dst;      // [nnz]  target state (eps/byte); return
                                      //        state in the caller (call)
  std::vector<int32_t> edge_callee;   // [nnz]  callee start state (call), else -1
};

// Token id -> raw bytes (no symbol classing -- PDA edges are byte ranges
// directly). Ragged like TokenSymbols: token t occupies
// bytes[offsets[t]:offsets[t+1]].
struct PdaTokens {
  int32_t vocab_size = 0;
  int32_t eos_id = -1;
  std::vector<int32_t> offsets;  // [vocab_size + 1]
  std::vector<uint8_t> bytes;    // [offsets.back()]
};

// A bounded set of stacks. `stack[i][0 .. depth[i)-1]` is config i's frames,
// top of stack last. `saturated` is set once any expansion had to be dropped
// for exceeding kPdaMaxConfigs / kPdaMaxDepth.
//
// Every field is int32_t (including the boolean `saturated`) so the struct is
// uniformly 4-byte aligned with no compiler-inserted padding: its layout is
// byte-identical to a flat int32 buffer of
// `2 + kPdaMaxConfigs + kPdaMaxConfigs*kPdaMaxDepth` elements in this field
// order. That is what lets a plain torch.int32 tensor be reinterpreted as
// `PdaConfigSet*` on the device side (bindings/torch_ops.cpp) without any
// host/device struct-layout assumptions beyond "no padding between int32s".
struct PdaConfigSet {
  int32_t num_configs = 0;
  int32_t saturated = 0;
  int32_t depth[kPdaMaxConfigs] = {};
  int32_t stack[kPdaMaxConfigs][kPdaMaxDepth] = {};
};

static_assert(sizeof(PdaConfigSet) ==
                 sizeof(int32_t) * (2 + kPdaMaxConfigs + kPdaMaxConfigs * kPdaMaxDepth),
             "PdaConfigSet must be padding-free for the flat torch.int32 tensor ABI");

// --- device-compatible primitives (header-only, shared by CPU and CUDA) ---

BPDECODE_HD inline bool pda_stack_eq(const int32_t* a, const int32_t* b, int32_t n) {
  for (int32_t i = 0; i < n; ++i) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

BPDECODE_HD inline bool pda_contains(const PdaConfigSet& cfg, const int32_t* st,
                                     int32_t depth) {
  for (int32_t i = 0; i < cfg.num_configs; ++i) {
    if (cfg.depth[i] == depth && pda_stack_eq(cfg.stack[i], st, depth)) return true;
  }
  return false;
}

// Returns true if appended, false if dropped (cap exceeded -> sets saturated).
BPDECODE_HD inline bool pda_push_config(PdaConfigSet& cfg, const int32_t* st,
                                        int32_t depth) {
  if (cfg.num_configs >= kPdaMaxConfigs) {
    cfg.saturated = 1;
    return false;
  }
  const int32_t i = cfg.num_configs++;
  cfg.depth[i] = depth;
  for (int32_t k = 0; k < depth; ++k) cfg.stack[i][k] = st[k];
  return true;
}

// Drop configs whose top state cannot reach its rule's accept.
template <class Table>
BPDECODE_HD inline void pda_prune_dead(const Table& g, PdaConfigSet& cfg) {
  int32_t out = 0;
  for (int32_t i = 0; i < cfg.num_configs; ++i) {
    const int32_t d = cfg.depth[i];
    if (d <= 0) continue;
    const int32_t top = cfg.stack[i][d - 1];
    if (!g.live[top]) continue;
    if (out != i) {
      cfg.depth[out] = d;
      for (int32_t k = 0; k < d; ++k) cfg.stack[out][k] = cfg.stack[i][k];
    }
    ++out;
  }
  cfg.num_configs = out;
}

// The functions below are templated on the table type so the exact same code
// runs against PdaTable (host, owns std::vector) and, from pda_cuda.cu,
// PdaTableView (device, raw pointers into an uploaded SoA buffer) -- both just
// need operator[] on `accept` / `live` / `edge_*` and `root_start` /
// `root_accept` members.

// Epsilon-closure to a fixpoint: resolves rule calls (push), rule completion
// (pop) and NFA epsilon edges. Mirrors PDA._close. Returns whether any config
// survives.
template <class Table>
BPDECODE_HD inline bool pda_close(const Table& g, PdaConfigSet& cfg) {
  bool changed = true;
  int32_t guard = 0;
  const int32_t max_rounds = kPdaMaxConfigs * kPdaMaxDepth + 8;
  while (changed && guard++ < max_rounds) {
    changed = false;
    const int32_t n = cfg.num_configs;
    for (int32_t i = 0; i < n; ++i) {
      const int32_t d = cfg.depth[i];
      if (d <= 0) continue;
      const int32_t top = cfg.stack[i][d - 1];

      if (g.accept[top] && d > 1) {
        if (!pda_contains(cfg, cfg.stack[i], d - 1)) {
          if (pda_push_config(cfg, cfg.stack[i], d - 1)) changed = true;
        }
      }

      const int32_t eb = g.edge_offsets[top];
      const int32_t ee = g.edge_offsets[top + 1];
      for (int32_t e = eb; e < ee; ++e) {
        const int32_t kind = g.edge_kind[e];
        if (kind == kPdaEps) {
          int32_t tmp[kPdaMaxDepth];
          for (int32_t k = 0; k < d - 1; ++k) tmp[k] = cfg.stack[i][k];
          tmp[d - 1] = g.edge_dst[e];
          if (!pda_contains(cfg, tmp, d)) {
            if (pda_push_config(cfg, tmp, d)) changed = true;
          }
        } else if (kind == kPdaCall) {
          if (d + 1 <= kPdaMaxDepth) {
            int32_t tmp[kPdaMaxDepth];
            for (int32_t k = 0; k < d - 1; ++k) tmp[k] = cfg.stack[i][k];
            tmp[d - 1] = g.edge_dst[e];      // return state, in the caller
            tmp[d] = g.edge_callee[e];       // push: callee start state
            if (!pda_contains(cfg, tmp, d + 1)) {
              if (pda_push_config(cfg, tmp, d + 1)) changed = true;
            }
          } else {
            cfg.saturated = 1;
          }
        }
      }
    }
  }
  pda_prune_dead(g, cfg);
  return cfg.num_configs > 0;
}

template <class Table>
BPDECODE_HD inline void pda_init(const Table& g, PdaConfigSet& cfg) {
  cfg.num_configs = 0;
  cfg.saturated = 0;
  int32_t st[1] = {g.root_start};
  pda_push_config(cfg, st, 1);
  pda_close(g, cfg);
}

// Consume one byte on the top frame of every config, then re-close. Returns
// whether the result is non-empty (the byte is on some valid path).
template <class Table>
BPDECODE_HD inline bool pda_advance_byte(const Table& g, PdaConfigSet& cfg,
                                         int32_t byte) {
  PdaConfigSet next;
  next.num_configs = 0;
  next.saturated = 0;
  for (int32_t i = 0; i < cfg.num_configs; ++i) {
    const int32_t d = cfg.depth[i];
    if (d <= 0) continue;
    const int32_t top = cfg.stack[i][d - 1];
    const int32_t eb = g.edge_offsets[top];
    const int32_t ee = g.edge_offsets[top + 1];
    for (int32_t e = eb; e < ee; ++e) {
      if (g.edge_kind[e] != kPdaByte) continue;
      if (byte < g.edge_lo[e] || byte > g.edge_hi[e]) continue;
      int32_t tmp[kPdaMaxDepth];
      for (int32_t k = 0; k < d - 1; ++k) tmp[k] = cfg.stack[i][k];
      tmp[d - 1] = g.edge_dst[e];
      if (!pda_contains(next, tmp, d)) pda_push_config(next, tmp, d);
    }
  }
  next.saturated = next.saturated || cfg.saturated;
  const bool ok = pda_close(g, next);
  cfg = next;
  return ok;
}

template <class Table>
BPDECODE_HD inline bool pda_is_complete(const Table& g, const PdaConfigSet& cfg) {
  for (int32_t i = 0; i < cfg.num_configs; ++i) {
    if (cfg.depth[i] == 1 && cfg.stack[i][0] == g.root_accept) return true;
  }
  return false;
}

// Simulate a whole token (its byte string) from a *copy* of `cfg`; does not
// mutate the caller's config. Returns whether the token is accepted.
template <class Table>
BPDECODE_HD inline bool pda_try_token(const Table& g, const PdaTokens& toks,
                                      PdaConfigSet cfg, int32_t token_id) {
  if (token_id == toks.eos_id) return pda_is_complete(g, cfg);
  const int32_t begin = toks.offsets[token_id];
  const int32_t end = toks.offsets[token_id + 1];
  for (int32_t i = begin; i < end; ++i) {
    if (!pda_advance_byte(g, cfg, toks.bytes[i])) return false;
  }
  return cfg.num_configs > 0;
}

// --- batched entry points (host declarations; CPU defs in pda_cpu.cpp) ---

// Packed allow-mask, one row (ceil(vocab/32) uint32 words) per request.
void compute_mask_pda_batch(const PdaTable& g, const PdaTokens& toks,
                            const PdaConfigSet* configs, int32_t batch,
                            uint32_t* out_bits);

// Fused mask + logit bias: logits is row-major [batch, vocab_size]; every
// token not allowed from configs[b] is set to neg_inf. Mirrors mask.hpp's
// apply_mask_batch for the FSA path.
void apply_mask_pda_batch(const PdaTable& g, const PdaTokens& toks,
                          const PdaConfigSet* configs, int32_t batch,
                          float* logits, float neg_inf);

// Advance each request's config-set by the token it sampled, in place.
// `ok[b]` is false if that token was off a valid path (config-set went empty);
// such a request's config-set is left unchanged (matches PDA/CFGConstraint,
// which raise rather than silently advancing into a dead state).
void advance_state_pda_batch(const PdaTable& g, const PdaTokens& toks,
                             PdaConfigSet* configs, const int32_t* token_ids,
                             int32_t batch, uint8_t* ok);

#ifdef BPDECODE_WITH_CUDA
// Device path: same contract, but every pointer is a device pointer and the
// grammar tables are already device-resident (see mask.hpp's CUDA section for
// the SoA upload convention this follows).
void compute_mask_pda_batch_cuda(
    int32_t num_states, int32_t root_start, int32_t root_accept,
    const uint8_t* accept, const uint8_t* live, const int32_t* edge_offsets,
    const int32_t* edge_kind, const int32_t* edge_lo, const int32_t* edge_hi,
    const int32_t* edge_dst, const int32_t* edge_callee, int32_t vocab_size,
    int32_t eos_id, const int32_t* tok_offsets, const uint8_t* tok_bytes,
    const PdaConfigSet* configs, int32_t batch, uint32_t* out_bits, void* stream);

void apply_mask_pda_batch_cuda(
    int32_t num_states, int32_t root_start, int32_t root_accept,
    const uint8_t* accept, const uint8_t* live, const int32_t* edge_offsets,
    const int32_t* edge_kind, const int32_t* edge_lo, const int32_t* edge_hi,
    const int32_t* edge_dst, const int32_t* edge_callee, int32_t vocab_size,
    int32_t eos_id, const int32_t* tok_offsets, const uint8_t* tok_bytes,
    const PdaConfigSet* configs, int32_t batch, float* logits, float neg_inf,
    void* stream);

void advance_state_pda_batch_cuda(
    int32_t num_states, int32_t root_start, int32_t root_accept,
    const uint8_t* accept, const uint8_t* live, const int32_t* edge_offsets,
    const int32_t* edge_kind, const int32_t* edge_lo, const int32_t* edge_hi,
    const int32_t* edge_dst, const int32_t* edge_callee, int32_t vocab_size,
    int32_t eos_id, const int32_t* tok_offsets, const uint8_t* tok_bytes,
    PdaConfigSet* configs, const int32_t* token_ids, int32_t batch, uint8_t* ok,
    void* stream);
#endif

}  // namespace bpdecode
