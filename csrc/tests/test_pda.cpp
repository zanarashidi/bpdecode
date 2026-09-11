#include "bpdecode/pda.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <string>
#include <vector>

using namespace bpdecode;

namespace {

// root ::= "(" root ")" | "x"   -- one self-recursive call.
//
// states: 0 start, 1 after eps into paren branch, 2 about to call root,
//         3 return point (about to match ")"), 4 accept, 5 the "x" branch.
//   0 --eps--> 1        0 --eps--> 5
//   1 --'('--> 2        2 --call(root@0, return=3)-->
//   3 --')'--> 4        5 --'x'--> 4
PdaTable make_paren_table() {
  PdaTable g;
  g.num_states = 6;
  g.root_start = 0;
  g.root_accept = 4;
  g.accept = {0, 0, 0, 0, 1, 0};
  g.live = {1, 1, 1, 1, 1, 1};
  g.edge_offsets = {0, 2, 3, 4, 5, 5, 6};
  g.edge_kind = {kPdaEps, kPdaEps, kPdaByte, kPdaCall, kPdaByte, kPdaByte};
  g.edge_lo = {0, 0, '(', 0, ')', 'x'};
  g.edge_hi = {0, 0, '(', 0, ')', 'x'};
  g.edge_dst = {1, 5, 2, 3, 4, 4};
  g.edge_callee = {-1, -1, -1, 0, -1, -1};
  return g;
}

// tokens: 0='(' 1=')' 2='x' 3=eos
PdaTokens make_paren_tokens() {
  PdaTokens t;
  t.vocab_size = 4;
  t.eos_id = 3;
  t.offsets = {0, 1, 2, 3, 3};
  t.bytes = {'(', ')', 'x'};
  return t;
}

bool feed(const PdaTable& g, PdaConfigSet& cfg, const std::string& s) {
  for (char c : s) {
    if (!pda_advance_byte(g, cfg, static_cast<uint8_t>(c))) return false;
  }
  return true;
}

}  // namespace

TEST(Pda, InitIsNotComplete) {
  auto g = make_paren_table();
  PdaConfigSet cfg;
  pda_init(g, cfg);
  EXPECT_FALSE(pda_is_complete(g, cfg));
}

TEST(Pda, AcceptsX) {
  auto g = make_paren_table();
  PdaConfigSet cfg;
  pda_init(g, cfg);
  ASSERT_TRUE(feed(g, cfg, "x"));
  EXPECT_TRUE(pda_is_complete(g, cfg));
}

TEST(Pda, RejectsY) {
  auto g = make_paren_table();
  PdaConfigSet cfg;
  pda_init(g, cfg);
  EXPECT_FALSE(feed(g, cfg, "y"));
}

TEST(Pda, RecursesOneLevel) {
  auto g = make_paren_table();
  PdaConfigSet cfg;
  pda_init(g, cfg);
  ASSERT_TRUE(feed(g, cfg, "("));
  EXPECT_FALSE(pda_is_complete(g, cfg));
  ASSERT_TRUE(feed(g, cfg, "x"));
  EXPECT_FALSE(pda_is_complete(g, cfg));  // still need the closing ')'
  ASSERT_TRUE(feed(g, cfg, ")"));
  EXPECT_TRUE(pda_is_complete(g, cfg));
}

TEST(Pda, RecursesTwoLevels) {
  auto g = make_paren_table();
  PdaConfigSet cfg;
  pda_init(g, cfg);
  EXPECT_TRUE(feed(g, cfg, "((x))"));
  EXPECT_TRUE(pda_is_complete(g, cfg));
}

TEST(Pda, UnbalancedParenIsIncomplete) {
  auto g = make_paren_table();
  PdaConfigSet cfg;
  pda_init(g, cfg);
  ASSERT_TRUE(feed(g, cfg, "(("));
  ASSERT_TRUE(feed(g, cfg, "x"));
  ASSERT_TRUE(feed(g, cfg, ")"));
  EXPECT_FALSE(pda_is_complete(g, cfg));  // one ')' short
}

TEST(Pda, TryTokenDoesNotMutate) {
  auto g = make_paren_table();
  auto toks = make_paren_tokens();
  PdaConfigSet cfg;
  pda_init(g, cfg);
  EXPECT_TRUE(pda_try_token(g, toks, cfg, 0));   // '(' -- byte-valid
  EXPECT_TRUE(pda_try_token(g, toks, cfg, 2));   // 'x' -- byte-valid
  EXPECT_FALSE(pda_is_complete(g, cfg));         // cfg itself untouched by try
}

TEST(Pda, ComputeMaskMatchesTryToken) {
  auto g = make_paren_table();
  auto toks = make_paren_tokens();
  PdaConfigSet cfg;
  pda_init(g, cfg);

  uint32_t bits = 0;
  compute_mask_pda_batch(g, toks, &cfg, 1, &bits);
  for (int32_t t = 0; t < toks.vocab_size; ++t) {
    const bool want = pda_try_token(g, toks, cfg, t);
    const bool got = (bits >> t) & 1u;
    EXPECT_EQ(got, want) << "token " << t;
  }
  // at the start: '(' and 'x' allowed, ')' and eos are not
  EXPECT_TRUE((bits >> 0) & 1u);
  EXPECT_FALSE((bits >> 1) & 1u);
  EXPECT_TRUE((bits >> 2) & 1u);
  EXPECT_FALSE((bits >> 3) & 1u);
}

TEST(Pda, ApplyMaskMatchesComputeMask) {
  auto g = make_paren_table();
  auto toks = make_paren_tokens();
  PdaConfigSet cfg;
  pda_init(g, cfg);

  uint32_t bits = 0;
  compute_mask_pda_batch(g, toks, &cfg, 1, &bits);

  std::vector<float> logits(toks.vocab_size, 1.0f);
  const float ninf = -std::numeric_limits<float>::infinity();
  apply_mask_pda_batch(g, toks, &cfg, 1, logits.data(), ninf);

  for (int32_t t = 0; t < toks.vocab_size; ++t) {
    const bool allowed_bits = (bits >> t) & 1u;
    const bool allowed_logits = logits[t] == 1.0f;
    EXPECT_EQ(allowed_bits, allowed_logits) << "token " << t;
  }
}

TEST(Pda, AdvanceStateBatchCommitsAndRejects) {
  auto g = make_paren_table();
  auto toks = make_paren_tokens();
  PdaConfigSet cfg;
  pda_init(g, cfg);

  int32_t tok = 0;   // '('
  uint8_t ok = 0;
  advance_state_pda_batch(g, toks, &cfg, &tok, 1, &ok);
  EXPECT_EQ(ok, 1);
  EXPECT_FALSE(pda_is_complete(g, cfg));

  tok = 1;  // ')' is not valid yet (need 'x' or '(' first)
  advance_state_pda_batch(g, toks, &cfg, &tok, 1, &ok);
  EXPECT_EQ(ok, 0);  // rejected, and cfg must be unchanged
  EXPECT_FALSE(pda_is_complete(g, cfg));

  tok = 2;  // 'x'
  advance_state_pda_batch(g, toks, &cfg, &tok, 1, &ok);
  EXPECT_EQ(ok, 1);

  tok = 1;  // ')'
  advance_state_pda_batch(g, toks, &cfg, &tok, 1, &ok);
  EXPECT_EQ(ok, 1);
  EXPECT_TRUE(pda_is_complete(g, cfg));

  tok = 3;  // eos, now that it's complete
  advance_state_pda_batch(g, toks, &cfg, &tok, 1, &ok);
  EXPECT_EQ(ok, 1);
}
