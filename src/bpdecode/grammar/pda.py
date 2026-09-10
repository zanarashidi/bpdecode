"""A config-set pushdown automaton over the rule NFAs.

The runtime state is a *set of stacks* -- one stack per way the input could be
parsed so far.  A stack frame is ``(rule_name, nfa_state)``; the bottom frame is
the root rule.  Epsilon-closure resolves rule calls (push), rule completion
(pop) and NFA epsilon moves to a fixpoint; :meth:`PDA.advance_byte` then
consumes a byte on the top frame of every stack.

This is the correct-but-unoptimised oracle (the analogue of
:class:`~bpdecode.reference.RegexConstraint` for CFGs).  Deep recursion is
capped; genuinely ambiguous grammars can blow the config-set cap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ir import Grammar
from .nfa import RuleNFA, compile_rules

Frame = tuple[str, int]
Config = tuple[Frame, ...]

DEFAULT_MAX_DEPTH = 64
DEFAULT_MAX_CONFIGS = 4096


class PDAOverflow(RuntimeError):
    """The config-set or stack depth exceeded its cap."""


def _rule_nonempty(rules: dict[str, RuleNFA]) -> dict[str, bool]:
    """Fixpoint: can each rule derive any string (empty included)?"""
    ok = dict.fromkeys(rules, False)
    changed = True
    while changed:
        changed = False
        for name, nfa in rules.items():
            if ok[name]:
                continue
            if _coreaches(nfa, nfa.accept, ok):
                ok[name] = True
                changed = True
    return ok


def _coreaches(nfa: RuleNFA, target: int, rule_ok: dict[str, bool]) -> bool:
    """Is ``target`` reachable from ``nfa.start`` over passable edges?"""
    seen = {nfa.start}
    stack = [nfa.start]
    while stack:
        s = stack.pop()
        if s == target:
            return True
        for label, dst in nfa.out(s):
            if label is not None and label[0] == "call" and not rule_ok.get(label[1]):
                continue
            if dst not in seen:
                seen.add(dst)
                stack.append(dst)
    return target in seen


def _coreachable_sets(
    rules: dict[str, RuleNFA], rule_ok: dict[str, bool]
) -> dict[str, frozenset[int]]:
    """Per rule: states from which the rule's accept state is reachable."""
    out: dict[str, frozenset[int]] = {}
    for name, nfa in rules.items():
        preds: dict[int, list[int]] = {}
        for s in range(nfa.num_states):
            for label, dst in nfa.out(s):
                if label is not None and label[0] == "call" and not rule_ok.get(
                    label[1]
                ):
                    continue
                preds.setdefault(dst, []).append(s)
        seen = {nfa.accept}
        stack = [nfa.accept]
        while stack:
            for p in preds.get(stack.pop(), ()):
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        out[name] = frozenset(seen)
    return out


@dataclass
class CompiledGrammar:
    rules: dict[str, RuleNFA]
    root: str
    coreachable: dict[str, frozenset[int]]
    # memos shared by every PDA over this grammar
    transition_memo: dict = field(default_factory=dict)  # closure + byte transition
    mask_memo: dict = field(default_factory=dict)  # (vocab id, config-set) -> token ids

    @classmethod
    def build(cls, grammar: Grammar) -> CompiledGrammar:
        rules = compile_rules(grammar)
        ok = _rule_nonempty(rules)
        return cls(rules, grammar.root, _coreachable_sets(rules, ok))


class PDA:
    def __init__(
        self,
        compiled: CompiledGrammar,
        *,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_configs: int = DEFAULT_MAX_CONFIGS,
    ) -> None:
        self.g = compiled
        self.max_depth = max_depth
        self.max_configs = max_configs
        # transitions are pure given the grammar; share the memo across PDAs of
        # the same compiled grammar so a fresh per-request PDA starts warm.
        self._memo = compiled.transition_memo
        start = self.g.rules[self.g.root].start
        self._configs: frozenset[Config] = self._close(
            frozenset({((self.g.root, start),)})
        )

    # --- state snapshot / restore -----------------------------------
    @property
    def configs(self) -> frozenset[Config]:
        return self._configs

    def set_configs(self, configs: frozenset[Config]) -> None:
        self._configs = configs

    def dead(self) -> bool:
        return not self._configs

    # --- epsilon closure -----------------------------------------------
    def _close(self, configs: frozenset[Config]) -> frozenset[Config]:
        cached = self._memo.get(("c", configs))
        if cached is not None:
            return cached
        result = self._close_uncached(configs)
        self._memo[("c", configs)] = result
        return result

    def _close_uncached(self, configs: frozenset[Config]) -> frozenset[Config]:
        out: set[Config] = set()
        work = list(configs)
        while work:
            cfg = work.pop()
            if cfg in out:
                continue
            out.add(cfg)
            if not cfg:
                continue
            rule, state = cfg[-1]
            nfa = self.g.rules[rule]
            # rule completion: pop
            if state == nfa.accept and len(cfg) > 1:
                work.append(cfg[:-1])
            for label, dst in nfa.out(state):
                if label is None:  # epsilon
                    work.append((*cfg[:-1], (rule, dst)))
                elif label[0] == "call":
                    if len(cfg) >= self.max_depth:
                        continue
                    callee = label[1]
                    work.append(
                        (*cfg[:-1], (rule, dst), (callee, self.g.rules[callee].start))
                    )
            if len(out) > self.max_configs:
                raise PDAOverflow(
                    f"config-set exceeded {self.max_configs}; grammar too ambiguous"
                )
        # prune configs whose top frame can no longer complete its rule
        return frozenset(
            cfg
            for cfg in out
            if not cfg or cfg[-1][1] in self.g.coreachable[cfg[-1][0]]
        )

    # --- stepping ----------------------------------------------------
    def advance_byte(self, b: int) -> bool:
        key = ("t", self._configs, b)
        nxt = self._memo.get(key)
        if nxt is None:
            raw: set[Config] = set()
            for cfg in self._configs:
                if not cfg:
                    continue
                rule, state = cfg[-1]
                for label, dst in self.g.rules[rule].out(state):
                    if (
                        label is not None
                        and label[0] == "byte"
                        and label[1] <= b <= label[2]
                    ):
                        raw.add((*cfg[:-1], (rule, dst)))
            nxt = self._close(frozenset(raw))
            self._memo[key] = nxt
        self._configs = nxt
        return bool(nxt)

    def first_byte_ranges(self) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        for cfg in self._configs:
            if not cfg:
                continue
            rule, state = cfg[-1]
            for label, _ in self.g.rules[rule].out(state):
                if label is not None and label[0] == "byte":
                    ranges.append((label[1], label[2]))
        return _merge(ranges)

    def is_complete(self) -> bool:
        root_accept = ((self.g.root, self.g.rules[self.g.root].accept),)
        return root_accept in self._configs


def _merge(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []
    ranges = sorted(ranges)
    out = [list(ranges[0])]
    for lo, hi in ranges[1:]:
        if lo <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [(lo, hi) for lo, hi in out]
