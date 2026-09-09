// Phase 1 lands the real batched kernel here: one warp per request, __ballot_sync
// to pack 32 tokens per uint32 word, transition table in shared/constant memory,
// overlapped with the model forward pass on a caller-supplied stream.
//
// For now this is a straight device port of the scalar reference so the CUDA
// build path stays green and the ABI is exercised end to end.
#include "bpdecode/mask.hpp"

namespace bpdecode {

void compute_mask_batch_cuda(const FsaTable& fsa, const TokenSymbols& toks,
                             const int32_t* states, int32_t batch,
                             uint32_t* out_bits, void* /*stream*/) {
  // TODO(phase-1): real device kernel. Fall back to host logic on the data as
  // uploaded; callers currently pass host-accessible (managed) pointers in tests.
  compute_mask_batch(fsa, toks, states, batch, out_bits);
}

}  // namespace bpdecode
