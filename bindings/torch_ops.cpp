// torch.ops.bpdecode.* -- thin glue from ATen tensors onto the bpdecode core.
//
// The FSA is passed as a bundle of tensors (the SoA / CSR upload form):
//   trans    int32  [num_states * num_symbols]
//   accept   uint8  [num_states]
//   live     uint8  [num_states]
//   offsets  int32  [vocab_size + 1]
//   symbols  int32  [offsets[-1]]
//   states   int32  [batch]
//
// CPU tensors are copied into FsaTable / TokenSymbols and run through the
// scalar core. CUDA tensors (when the core was built with CUDA) are handed to
// the device kernels by raw pointer, no copy. All tensors for one call must be
// on the same device.

#include <Python.h>
#include <torch/library.h>
#include <torch/torch.h>

#include <cstring>
#include <vector>

#include "bpdecode/mask.hpp"
#include "bpdecode/pda.hpp"

#if defined(BPDECODE_WITH_CUDA)
#include <c10/cuda/CUDAStream.h>
#endif

namespace bpdecode {
namespace {

// The mask kernels launch on torch's current CUDA stream and do not sync, so
// they overlap the forward pass; ordering with the sampling that reads `logits`
// is guaranteed because that runs on the same stream.
inline void* current_cuda_stream() {
#if defined(BPDECODE_WITH_CUDA)
  return c10::cuda::getCurrentCUDAStream().stream();
#else
  return nullptr;
#endif
}

void check_cpu_layout(const at::Tensor& trans, const at::Tensor& accept,
                      const at::Tensor& live, const at::Tensor& offsets,
                      const at::Tensor& symbols) {
  TORCH_CHECK(trans.dtype() == at::kInt, "trans must be int32");
  TORCH_CHECK(accept.dtype() == at::kByte, "accept must be uint8");
  TORCH_CHECK(live.dtype() == at::kByte, "live must be uint8");
  TORCH_CHECK(offsets.dtype() == at::kInt, "offsets must be int32");
  TORCH_CHECK(symbols.dtype() == at::kInt, "symbols must be int32");
}

FsaTable fsa_from_cpu(const at::Tensor& trans, const at::Tensor& accept,
                      const at::Tensor& live, int64_t num_symbols,
                      int64_t dead) {
  auto tr = trans.contiguous();
  auto ac = accept.contiguous();
  auto lv = live.contiguous();
  FsaTable f;
  f.num_symbols = static_cast<int32_t>(num_symbols);
  f.num_states = static_cast<int32_t>(ac.numel());
  f.start = 0;
  f.dead = static_cast<int32_t>(dead);
  const auto* trp = tr.data_ptr<int32_t>();
  const auto* acp = ac.data_ptr<uint8_t>();
  const auto* lvp = lv.data_ptr<uint8_t>();
  f.trans.assign(trp, trp + tr.numel());
  f.accept.assign(acp, acp + ac.numel());
  f.live.assign(lvp, lvp + lv.numel());
  return f;
}

TokenSymbols toks_from_cpu(const at::Tensor& offsets, const at::Tensor& symbols,
                           int64_t eos_id) {
  auto of = offsets.contiguous();
  auto sy = symbols.contiguous();
  TokenSymbols t;
  t.vocab_size = static_cast<int32_t>(of.numel() - 1);
  t.eos_id = static_cast<int32_t>(eos_id);
  const auto* ofp = of.data_ptr<int32_t>();
  const auto* syp = sy.data_ptr<int32_t>();
  t.offsets.assign(ofp, ofp + of.numel());
  t.symbols.assign(syp, syp + sy.numel());
  return t;
}

at::Tensor build_reachability_op(const at::Tensor& trans,
                                 const at::Tensor& accept, int64_t num_symbols) {
  TORCH_CHECK(!trans.is_cuda(), "build_reachability is host-only (run it once at compile time)");
  TORCH_CHECK(trans.dtype() == at::kInt && accept.dtype() == at::kByte,
              "trans int32, accept uint8");
  auto ac = accept.contiguous().cpu();
  auto tr = trans.contiguous().cpu();
  FsaTable f;
  f.num_symbols = static_cast<int32_t>(num_symbols);
  f.num_states = static_cast<int32_t>(ac.numel());
  f.dead = f.num_states - 1;
  const auto* trp = tr.data_ptr<int32_t>();
  const auto* acp = ac.data_ptr<uint8_t>();
  f.trans.assign(trp, trp + tr.numel());
  f.accept.assign(acp, acp + ac.numel());
  build_reachability(f);
  auto out = at::empty({f.num_states}, at::kByte);
  std::copy(f.live.begin(), f.live.end(), out.data_ptr<uint8_t>());
  return out;
}

at::Tensor apply_mask_op(at::Tensor logits, const at::Tensor& trans,
                         const at::Tensor& accept, const at::Tensor& live,
                         int64_t num_symbols, int64_t dead,
                         const at::Tensor& offsets, const at::Tensor& symbols,
                         int64_t eos_id, const at::Tensor& states,
                         double neg_inf) {
  TORCH_CHECK(logits.dim() == 2 && logits.dtype() == at::kFloat,
              "logits must be float32 [batch, vocab]");
  TORCH_CHECK(states.dtype() == at::kInt, "states must be int32");
  const int32_t batch = static_cast<int32_t>(logits.size(0));
  const int32_t vocab = static_cast<int32_t>(logits.size(1));
  TORCH_CHECK(states.numel() == batch, "one state per logits row");
  logits = logits.contiguous();
  auto st = states.contiguous();

  if (logits.is_cuda()) {
#ifdef BPDECODE_WITH_CUDA
    TORCH_CHECK(trans.is_cuda() && accept.is_cuda() && live.is_cuda() &&
                    offsets.is_cuda() && symbols.is_cuda() && states.is_cuda(),
                "for a CUDA call every FSA tensor must be on the same device");
    apply_mask_batch_cuda(
        static_cast<int32_t>(accept.numel()), static_cast<int32_t>(num_symbols),
        trans.contiguous().data_ptr<int32_t>(),
        accept.contiguous().data_ptr<uint8_t>(),
        live.contiguous().data_ptr<uint8_t>(), static_cast<int32_t>(dead),
        offsets.contiguous().data_ptr<int32_t>(),
        symbols.contiguous().data_ptr<int32_t>(), vocab,
        static_cast<int32_t>(eos_id), st.data_ptr<int32_t>(), batch,
        logits.data_ptr<float>(), static_cast<float>(neg_inf),
        /*stream=*/current_cuda_stream());
    return logits;
#else
    TORCH_CHECK(false, "bpdecode was built without CUDA support");
#endif
  }

  check_cpu_layout(trans, accept, live, offsets, symbols);
  FsaTable f = fsa_from_cpu(trans, accept, live, num_symbols, dead);
  TokenSymbols t = toks_from_cpu(offsets, symbols, eos_id);
  TORCH_CHECK(t.vocab_size == vocab, "offsets imply a different vocab size");
  apply_mask_batch(f, t, st.data_ptr<int32_t>(), batch, logits.data_ptr<float>(),
                   static_cast<float>(neg_inf));
  return logits;
}

at::Tensor compute_mask_op(const at::Tensor& trans, const at::Tensor& accept,
                           const at::Tensor& live, int64_t num_symbols,
                           int64_t dead, const at::Tensor& offsets,
                           const at::Tensor& symbols, int64_t eos_id,
                           const at::Tensor& states) {
  const int32_t batch = static_cast<int32_t>(states.numel());
  const int32_t vocab = static_cast<int32_t>(offsets.numel() - 1);
  const int32_t words = (vocab + 31) / 32;
  auto st = states.contiguous();
  auto out = at::empty({batch, words},
                       states.options().dtype(at::kInt).device(states.device()));
  auto* out_bits = reinterpret_cast<uint32_t*>(out.data_ptr<int32_t>());

  if (states.is_cuda()) {
#ifdef BPDECODE_WITH_CUDA
    compute_mask_batch_cuda(
        static_cast<int32_t>(accept.numel()), static_cast<int32_t>(num_symbols),
        trans.contiguous().data_ptr<int32_t>(),
        accept.contiguous().data_ptr<uint8_t>(),
        live.contiguous().data_ptr<uint8_t>(), static_cast<int32_t>(dead),
        offsets.contiguous().data_ptr<int32_t>(),
        symbols.contiguous().data_ptr<int32_t>(), vocab,
        static_cast<int32_t>(eos_id), st.data_ptr<int32_t>(), batch, out_bits,
        /*stream=*/current_cuda_stream());
    return out;
#else
    TORCH_CHECK(false, "bpdecode was built without CUDA support");
#endif
  }

  check_cpu_layout(trans, accept, live, offsets, symbols);
  FsaTable f = fsa_from_cpu(trans, accept, live, num_symbols, dead);
  TokenSymbols t = toks_from_cpu(offsets, symbols, eos_id);
  compute_mask_batch(f, t, st.data_ptr<int32_t>(), batch, out_bits);
  return out;
}

at::Tensor advance_state_op(const at::Tensor& trans, const at::Tensor& accept,
                            const at::Tensor& live, int64_t num_symbols,
                            int64_t dead, const at::Tensor& offsets,
                            const at::Tensor& symbols, int64_t eos_id,
                            const at::Tensor& states,
                            const at::Tensor& token_ids) {
  TORCH_CHECK(states.dtype() == at::kInt && token_ids.dtype() == at::kInt,
              "states / token_ids must be int32");
  const int32_t batch = static_cast<int32_t>(states.numel());
  TORCH_CHECK(token_ids.numel() == batch, "one token id per state");
  auto st = states.contiguous();
  auto tk = token_ids.contiguous();
  auto out = at::empty({batch}, states.options());

  if (states.is_cuda()) {
#ifdef BPDECODE_WITH_CUDA
    advance_state_batch_cuda(
        static_cast<int32_t>(accept.numel()), static_cast<int32_t>(num_symbols),
        trans.contiguous().data_ptr<int32_t>(),
        accept.contiguous().data_ptr<uint8_t>(),
        live.contiguous().data_ptr<uint8_t>(), static_cast<int32_t>(dead),
        offsets.contiguous().data_ptr<int32_t>(),
        symbols.contiguous().data_ptr<int32_t>(),
        static_cast<int32_t>(offsets.numel() - 1), static_cast<int32_t>(eos_id),
        st.data_ptr<int32_t>(), tk.data_ptr<int32_t>(), batch,
        out.data_ptr<int32_t>(), /*stream=*/current_cuda_stream());
    return out;
#else
    TORCH_CHECK(false, "bpdecode was built without CUDA support");
#endif
  }

  check_cpu_layout(trans, accept, live, offsets, symbols);
  FsaTable f = fsa_from_cpu(trans, accept, live, num_symbols, dead);
  TokenSymbols t = toks_from_cpu(offsets, symbols, eos_id);
  advance_state_batch(f, t, st.data_ptr<int32_t>(), tk.data_ptr<int32_t>(), batch,
                      out.data_ptr<int32_t>());
  return out;
}

// --- PDA (CFG) ops -------------------------------------------------------
//
// The grammar (a flattened CompiledGrammar -- see bpdecode.grammar.device) is
// a bundle of tensors, same spirit as the FSA bundle above:
//   accept/live         uint8  [num_states]
//   edge_offsets        int32  [num_states + 1]
//   edge_kind/lo/hi/dst/callee  int32  [nnz]
//   root_start/root_accept      python ints
//   tok_offsets         int32  [vocab_size + 1]
//   tok_bytes           uint8  [tok_offsets[-1]]
// A config-set is a flat int32 tensor of kPdaConfigFlat elements per request
// (`[batch, kPdaConfigFlat]`) -- PdaConfigSet is deliberately padding-free
// (see pda.hpp) so this is reinterpret_cast-compatible with PdaConfigSet* on
// both CPU and CUDA, no copy needed for the config-set itself.

constexpr int64_t kPdaConfigFlat =
    2 + kPdaMaxConfigs + static_cast<int64_t>(kPdaMaxConfigs) * kPdaMaxDepth;

PdaTable pda_table_from_cpu(const at::Tensor& accept, const at::Tensor& live,
                            const at::Tensor& edge_offsets,
                            const at::Tensor& edge_kind, const at::Tensor& edge_lo,
                            const at::Tensor& edge_hi, const at::Tensor& edge_dst,
                            const at::Tensor& edge_callee, int64_t root_start,
                            int64_t root_accept) {
  auto ac = accept.contiguous().cpu();
  auto lv = live.contiguous().cpu();
  auto eo = edge_offsets.contiguous().cpu();
  auto ek = edge_kind.contiguous().cpu();
  auto elo = edge_lo.contiguous().cpu();
  auto ehi = edge_hi.contiguous().cpu();
  auto ed = edge_dst.contiguous().cpu();
  auto ec = edge_callee.contiguous().cpu();
  PdaTable g;
  g.num_states = static_cast<int32_t>(ac.numel());
  g.root_start = static_cast<int32_t>(root_start);
  g.root_accept = static_cast<int32_t>(root_accept);
  const auto* acp = ac.data_ptr<uint8_t>();
  const auto* lvp = lv.data_ptr<uint8_t>();
  g.accept.assign(acp, acp + ac.numel());
  g.live.assign(lvp, lvp + lv.numel());
  const auto* eop = eo.data_ptr<int32_t>();
  g.edge_offsets.assign(eop, eop + eo.numel());
  const auto* ekp = ek.data_ptr<int32_t>();
  g.edge_kind.assign(ekp, ekp + ek.numel());
  const auto* elop = elo.data_ptr<int32_t>();
  g.edge_lo.assign(elop, elop + elo.numel());
  const auto* ehip = ehi.data_ptr<int32_t>();
  g.edge_hi.assign(ehip, ehip + ehi.numel());
  const auto* edp = ed.data_ptr<int32_t>();
  g.edge_dst.assign(edp, edp + ed.numel());
  const auto* ecp = ec.data_ptr<int32_t>();
  g.edge_callee.assign(ecp, ecp + ec.numel());
  return g;
}

PdaTokens pda_tokens_from_cpu(const at::Tensor& tok_offsets,
                              const at::Tensor& tok_bytes, int64_t eos_id) {
  auto to = tok_offsets.contiguous().cpu();
  auto tb = tok_bytes.contiguous().cpu();
  PdaTokens t;
  t.vocab_size = static_cast<int32_t>(to.numel() - 1);
  t.eos_id = static_cast<int32_t>(eos_id);
  const auto* top = to.data_ptr<int32_t>();
  t.offsets.assign(top, top + to.numel());
  const auto* tbp = tb.data_ptr<uint8_t>();
  t.bytes.assign(tbp, tbp + tb.numel());
  return t;
}

void pda_check_configs(const at::Tensor& configs) {
  TORCH_CHECK(configs.dtype() == at::kInt, "configs must be int32");
  TORCH_CHECK(configs.dim() == 2 && configs.size(1) == kPdaConfigFlat,
              "configs must be [batch, ", kPdaConfigFlat, "]");
  TORCH_CHECK(configs.is_contiguous(), "configs must be contiguous");
}

at::Tensor pda_init_op(const at::Tensor& accept, const at::Tensor& live,
                       const at::Tensor& edge_offsets, const at::Tensor& edge_kind,
                       const at::Tensor& edge_lo, const at::Tensor& edge_hi,
                       const at::Tensor& edge_dst, const at::Tensor& edge_callee,
                       int64_t root_start, int64_t root_accept) {
  PdaTable g = pda_table_from_cpu(accept, live, edge_offsets, edge_kind, edge_lo,
                                  edge_hi, edge_dst, edge_callee, root_start,
                                  root_accept);
  PdaConfigSet cfg;
  pda_init(g, cfg);
  auto out = at::empty({kPdaConfigFlat}, at::kInt);
  static_assert(sizeof(PdaConfigSet) == sizeof(int32_t) * kPdaConfigFlat, "");
  std::memcpy(out.data_ptr<int32_t>(), &cfg, sizeof(PdaConfigSet));
  return out;
}

at::Tensor pda_apply_mask_op(at::Tensor logits, const at::Tensor& accept,
                             const at::Tensor& live, const at::Tensor& edge_offsets,
                             const at::Tensor& edge_kind, const at::Tensor& edge_lo,
                             const at::Tensor& edge_hi, const at::Tensor& edge_dst,
                             const at::Tensor& edge_callee, int64_t root_start,
                             int64_t root_accept, const at::Tensor& tok_offsets,
                             const at::Tensor& tok_bytes, int64_t eos_id,
                             const at::Tensor& configs, double neg_inf) {
  TORCH_CHECK(logits.dim() == 2 && logits.dtype() == at::kFloat,
              "logits must be float32 [batch, vocab]");
  pda_check_configs(configs);
  const int32_t batch = static_cast<int32_t>(logits.size(0));
  const int32_t vocab = static_cast<int32_t>(logits.size(1));
  TORCH_CHECK(configs.size(0) == batch, "one config-set per logits row");
  logits = logits.contiguous();

  if (logits.is_cuda()) {
#ifdef BPDECODE_WITH_CUDA
    TORCH_CHECK(accept.is_cuda() && live.is_cuda() && edge_offsets.is_cuda() &&
                    tok_offsets.is_cuda() && configs.is_cuda(),
                "for a CUDA call every grammar/config tensor must be on the same device");
    apply_mask_pda_batch_cuda(
        static_cast<int32_t>(accept.numel()), static_cast<int32_t>(root_start),
        static_cast<int32_t>(root_accept), accept.contiguous().data_ptr<uint8_t>(),
        live.contiguous().data_ptr<uint8_t>(),
        edge_offsets.contiguous().data_ptr<int32_t>(),
        edge_kind.contiguous().data_ptr<int32_t>(),
        edge_lo.contiguous().data_ptr<int32_t>(),
        edge_hi.contiguous().data_ptr<int32_t>(),
        edge_dst.contiguous().data_ptr<int32_t>(),
        edge_callee.contiguous().data_ptr<int32_t>(), vocab,
        static_cast<int32_t>(eos_id), tok_offsets.contiguous().data_ptr<int32_t>(),
        tok_bytes.contiguous().data_ptr<uint8_t>(),
        reinterpret_cast<const PdaConfigSet*>(configs.data_ptr<int32_t>()), batch,
        logits.data_ptr<float>(), static_cast<float>(neg_inf), current_cuda_stream());
    return logits;
#else
    TORCH_CHECK(false, "bpdecode was built without CUDA support");
#endif
  }

  PdaTable g = pda_table_from_cpu(accept, live, edge_offsets, edge_kind, edge_lo,
                                  edge_hi, edge_dst, edge_callee, root_start,
                                  root_accept);
  PdaTokens t = pda_tokens_from_cpu(tok_offsets, tok_bytes, eos_id);
  TORCH_CHECK(t.vocab_size == vocab, "tok_offsets imply a different vocab size");
  const auto* cfgs = reinterpret_cast<const PdaConfigSet*>(configs.data_ptr<int32_t>());
  apply_mask_pda_batch(g, t, cfgs, batch, logits.data_ptr<float>(),
                       static_cast<float>(neg_inf));
  return logits;
}

at::Tensor pda_advance_state_op(
    at::Tensor configs, const at::Tensor& accept, const at::Tensor& live,
    const at::Tensor& edge_offsets, const at::Tensor& edge_kind,
    const at::Tensor& edge_lo, const at::Tensor& edge_hi, const at::Tensor& edge_dst,
    const at::Tensor& edge_callee, int64_t root_start, int64_t root_accept,
    const at::Tensor& tok_offsets, const at::Tensor& tok_bytes, int64_t eos_id,
    const at::Tensor& token_ids) {
  pda_check_configs(configs);
  const int32_t batch = static_cast<int32_t>(configs.size(0));
  TORCH_CHECK(token_ids.dtype() == at::kInt && token_ids.numel() == batch,
              "token_ids must be int32, one per config-set");
  auto tk = token_ids.contiguous();
  auto ok = at::empty({batch}, configs.options().dtype(at::kByte));

  if (configs.is_cuda()) {
#ifdef BPDECODE_WITH_CUDA
    TORCH_CHECK(accept.is_cuda() && live.is_cuda() && edge_offsets.is_cuda() &&
                    tok_offsets.is_cuda() && token_ids.is_cuda(),
                "for a CUDA call every grammar/config tensor must be on the same device");
    advance_state_pda_batch_cuda(
        static_cast<int32_t>(accept.numel()), static_cast<int32_t>(root_start),
        static_cast<int32_t>(root_accept), accept.contiguous().data_ptr<uint8_t>(),
        live.contiguous().data_ptr<uint8_t>(),
        edge_offsets.contiguous().data_ptr<int32_t>(),
        edge_kind.contiguous().data_ptr<int32_t>(),
        edge_lo.contiguous().data_ptr<int32_t>(),
        edge_hi.contiguous().data_ptr<int32_t>(),
        edge_dst.contiguous().data_ptr<int32_t>(),
        edge_callee.contiguous().data_ptr<int32_t>(), /*vocab_size=*/0,
        static_cast<int32_t>(eos_id), tok_offsets.contiguous().data_ptr<int32_t>(),
        tok_bytes.contiguous().data_ptr<uint8_t>(),
        reinterpret_cast<PdaConfigSet*>(configs.data_ptr<int32_t>()),
        tk.data_ptr<int32_t>(), batch, ok.data_ptr<uint8_t>(), current_cuda_stream());
    return ok;
#else
    TORCH_CHECK(false, "bpdecode was built without CUDA support");
#endif
  }

  PdaTable g = pda_table_from_cpu(accept, live, edge_offsets, edge_kind, edge_lo,
                                  edge_hi, edge_dst, edge_callee, root_start,
                                  root_accept);
  PdaTokens t = pda_tokens_from_cpu(tok_offsets, tok_bytes, eos_id);
  auto* cfgs = reinterpret_cast<PdaConfigSet*>(configs.data_ptr<int32_t>());
  advance_state_pda_batch(g, t, cfgs, tk.data_ptr<int32_t>(), batch,
                          ok.data_ptr<uint8_t>());
  return ok;
}

TORCH_LIBRARY(bpdecode, m) {
  m.def(
      "build_reachability(Tensor trans, Tensor accept, int num_symbols) -> Tensor");
  m.def(
      "apply_mask_(Tensor(a!) logits, Tensor trans, Tensor accept, Tensor live, "
      "int num_symbols, int dead, Tensor offsets, Tensor symbols, int eos_id, "
      "Tensor states, float neg_inf) -> Tensor(a!)");
  m.def(
      "compute_mask(Tensor trans, Tensor accept, Tensor live, int num_symbols, "
      "int dead, Tensor offsets, Tensor symbols, int eos_id, Tensor states) "
      "-> Tensor");
  m.def(
      "advance_state(Tensor trans, Tensor accept, Tensor live, int num_symbols, "
      "int dead, Tensor offsets, Tensor symbols, int eos_id, Tensor states, "
      "Tensor token_ids) -> Tensor");

  m.def(
      "pda_init(Tensor accept, Tensor live, Tensor edge_offsets, Tensor edge_kind, "
      "Tensor edge_lo, Tensor edge_hi, Tensor edge_dst, Tensor edge_callee, "
      "int root_start, int root_accept) -> Tensor");
  m.def(
      "pda_apply_mask_(Tensor(a!) logits, Tensor accept, Tensor live, "
      "Tensor edge_offsets, Tensor edge_kind, Tensor edge_lo, Tensor edge_hi, "
      "Tensor edge_dst, Tensor edge_callee, int root_start, int root_accept, "
      "Tensor tok_offsets, Tensor tok_bytes, int eos_id, Tensor configs, "
      "float neg_inf) -> Tensor(a!)");
  m.def(
      "pda_advance_state(Tensor(a!) configs, Tensor accept, Tensor live, "
      "Tensor edge_offsets, Tensor edge_kind, Tensor edge_lo, Tensor edge_hi, "
      "Tensor edge_dst, Tensor edge_callee, int root_start, int root_accept, "
      "Tensor tok_offsets, Tensor tok_bytes, int eos_id, Tensor token_ids) "
      "-> Tensor");
}

TORCH_LIBRARY_IMPL(bpdecode, CompositeExplicitAutograd, m) {
  m.impl("build_reachability", TORCH_FN(build_reachability_op));
  m.impl("apply_mask_", TORCH_FN(apply_mask_op));
  m.impl("compute_mask", TORCH_FN(compute_mask_op));
  m.impl("advance_state", TORCH_FN(advance_state_op));
  m.impl("pda_init", TORCH_FN(pda_init_op));
  m.impl("pda_apply_mask_", TORCH_FN(pda_apply_mask_op));
  m.impl("pda_advance_state", TORCH_FN(pda_advance_state_op));
}

}  // namespace
}  // namespace bpdecode

// The op registration runs from static initializers on dlopen; this module
// object just gives CPython something to import (`import bpdecode._C`).
static struct PyModuleDef bpdecode_C_module = {
    PyModuleDef_HEAD_INIT, "_C", "bpdecode native ops", -1, nullptr,
};

extern "C" PyObject* PyInit__C() { return PyModule_Create(&bpdecode_C_module); }
