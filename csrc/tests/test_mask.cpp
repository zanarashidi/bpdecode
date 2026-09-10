#include "bpdecode/mask.hpp"

#include <gtest/gtest.h>

using namespace bpdecode;

namespace {

// DFA for the language "ab" over 2 symbol classes: 0 = 'a', 1 = 'b'.
//   s0 --a--> s1 --b--> s2(accept);  everything else -> dead (s3)
FsaTable make_ab_fsa() {
  FsaTable f;
  f.num_states = 4;
  f.num_symbols = 2;
  f.start = 0;
  f.dead = 3;
  f.trans = {
      /*s0*/ 1, 3,
      /*s1*/ 3, 2,
      /*s2*/ 3, 3,
      /*s3*/ 3, 3,
  };
  f.accept = {0, 0, 1, 0};
  f.live = {1, 1, 1, 0};
  return f;
}

// vocab: 0="a"(sym 0), 1="b"(sym 1), 2="ab"(sym 0,1), 3="c"(no class -> -1), 4=EOS
TokenSymbols make_toks() {
  TokenSymbols t;
  t.vocab_size = 5;
  t.eos_id = 4;
  t.offsets = {0, 1, 2, 4, 5, 5};
  t.symbols = {0, 1, 0, 1, -1};
  return t;
}

}  // namespace

TEST(Reachability, MarksOnlyStatesThatCanAccept) {
  auto f = make_ab_fsa();
  f.live.clear();
  build_reachability(f);
  ASSERT_EQ(f.live.size(), 4u);
  EXPECT_EQ(f.live[0], 1);  // s0 -a-> s1 -b-> s2(accept)
  EXPECT_EQ(f.live[1], 1);
  EXPECT_EQ(f.live[2], 1);  // accepting
  EXPECT_EQ(f.live[3], 0);  // dead
}

TEST(Reachability, DeadEndBranchIsNotLive) {
  // s0 --0--> s1(accept);  s0 --1--> s2 --*--> s2 (trap, never accepts)
  FsaTable f;
  f.num_states = 3;
  f.num_symbols = 2;
  f.start = 0;
  f.dead = 2;
  f.trans = {1, 2, 1, 1, 2, 2};
  f.accept = {0, 1, 0};
  build_reachability(f);
  EXPECT_EQ(f.live[0], 1);
  EXPECT_EQ(f.live[1], 1);
  EXPECT_EQ(f.live[2], 0);
}

TEST(Step, WalksAndRejects) {
  auto f = make_ab_fsa();
  auto t = make_toks();
  EXPECT_EQ(step(f, t, 0, 0), 1);   // 'a' from start
  EXPECT_EQ(step(f, t, 0, 1), -1);  // 'b' from start -> dead
  EXPECT_EQ(step(f, t, 1, 1), 2);   // 'b' from s1 -> accept
  EXPECT_EQ(step(f, t, 0, 2), 2);   // "ab" in one token
  EXPECT_EQ(step(f, t, 0, 3), -1);  // unknown symbol
}

TEST(Step, EosOnlyWhenAccepting) {
  auto f = make_ab_fsa();
  auto t = make_toks();
  EXPECT_EQ(step(f, t, 0, 4), -1);
  EXPECT_EQ(step(f, t, 2, 4), 2);
}

TEST(Mask, BitsMatchStep) {
  auto f = make_ab_fsa();
  auto t = make_toks();
  uint32_t bits = 0;
  compute_mask(f, t, 0, &bits);
  EXPECT_TRUE(bits & (1u << 0));   // "a"
  EXPECT_TRUE(bits & (1u << 2));   // "ab"
  EXPECT_FALSE(bits & (1u << 1));  // "b"
  EXPECT_FALSE(bits & (1u << 4));  // EOS (start not accepting)

  bits = 0;
  compute_mask(f, t, 2, &bits);
  EXPECT_TRUE(bits & (1u << 4));  // EOS allowed at accept state
}

TEST(MaskBatch, IndependentRows) {
  auto f = make_ab_fsa();
  auto t = make_toks();
  int32_t states[2] = {0, 2};
  uint32_t out[2] = {0, 0};
  compute_mask_batch(f, t, states, 2, out);
  EXPECT_TRUE(out[0] & (1u << 0));
  EXPECT_FALSE(out[0] & (1u << 4));
  EXPECT_TRUE(out[1] & (1u << 4));
}
