"""Regression test: the bare-family prompts must contain the goal text.

Commit daec8f6 dropped the ``+ goal`` suffix from ``bare_prompt_for`` /
``soft_gate_prompt_for``, so every bare / soft-gate / self-verify benchmark run
asked the model "Goal: " with no goal after it. The model had no idea what to
design and emitted the same default chip for every goal — a confound that
invalidated the memory-system numbers. The goal text must be in the prompt, and
the only place a future regression can hide is inside these two functions, so
assert it directly.
"""

from __future__ import annotations

from eval.benchmark import bare_prompt_for, load_gold, soft_gate_prompt_for


def test_bare_prompt_contains_goal():
    gold = load_gold()[0]
    prompt = bare_prompt_for(gold)
    assert gold.goal in prompt, "bare prompt must contain the goal text"
    # The old regression ended the prompt at "Goal: " with nothing after it.
    assert not prompt.rstrip().endswith("Goal:"), "prompt must not end at 'Goal:'"


def test_soft_gate_prompt_contains_goal():
    gold = load_gold()[0]
    prompt = soft_gate_prompt_for(gold)
    assert gold.goal in prompt, "soft-gate prompt must contain the goal text"
    assert not prompt.rstrip().endswith("Goal:"), "prompt must not end at 'Goal:'"


def test_all_gold_goals_reach_the_prompt():
    gold = load_gold()
    blind = load_gold("eval/gold_blind.json")
    for g in gold + blind:
        assert g.goal in bare_prompt_for(g), f"bare prompt missing goal for {g.id}"
        assert g.goal in soft_gate_prompt_for(g), f"soft-gate prompt missing goal for {g.id}"
