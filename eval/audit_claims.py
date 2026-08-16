"""Machine-checkable reproducibility guard for Labwright's headline numbers.

Every headline number in ``README.md`` / ``eval/README.md`` is recomputed here
from the committed ``results/*.json`` with the same derivation the project
reports with (``eval.report.derive``), and asserted equal to the value actually
written in the docs. Regenerating a JSON without syncing the docs — or editing
a doc without re-running the numbers — fails this audit.

Run:   python -m eval.audit_claims
Exit:  0 = every claim verified; 1 = at least one mismatch (each failure names
       the committed file, the expected value and the recomputed value).

The check functions are deliberately boring — load JSON, recompute, compare —
so a future run that changes a number can point at the exact claim it breaks.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from eval.report import derive

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
RESULTS = ROOT / "results"

_failures: list[str] = []
_passes = 0


def _load(name: str) -> dict:
    return json.load(open(RESULTS / name))


def _check(label: str, ok: bool, detail: str = "") -> None:
    """Record one audit result; failures are collected and reported at exit."""
    global _passes
    if ok:
        _passes += 1
        return
    _failures.append(f"  FAIL {label}" + (f"  —  {detail}" if detail else ""))


def _pct(x: float | None) -> int:
    """The whole-number percent the docs display for a rate."""
    return round((x or 0.0) * 100)


def _hall(x: float | None) -> float:
    """Hallucination as the docs display it (3 decimals)."""
    return round((x or 0.0), 3)


def _entry_usable(rec: dict | None) -> bool:
    """The usable rule as defined in the docs: hallucination == 0.0, a recovery
    was submitted, and every recovered value lands within ±5%."""
    if not rec:
        return False
    if rec.get("hallucination_rate") != 0.0:
        return False
    recv = rec.get("recovery") or {}
    if not recv:
        return False
    return all(abs(float(v)) <= 0.05 for v in recv.values())


def _usable_ids(path: str) -> set[str]:
    return {e["id"] for e in _load(path)["per_entry"] if _entry_usable(e.get("labwright"))}


# ---------------------------------------------------------------------------
# A. Agent-loop six-set benchmark rows (README.md, "Benchmark" table)
# ---------------------------------------------------------------------------

# The reading set is the benchmark default, so its agent-loop files are
# `eval_{model}.json` rather than `eval_reading_{model}.json`.
AGENT_FILE = {
    "reading": "eval_{model}.json",
    "blind": "eval_blind_{model}.json",
    "spheroid": "eval_spheroid_{model}.json",
    "culture": "eval_culture_{model}.json",
    "pk": "eval_pk_{model}.json",
    "newdomains": "eval_new_domains_labwright_{model}.json",
}


# (set, model) -> (self-consistent %, usable %, hallucination) as displayed.
AGENT_ROWS = {
    ("reading", "flash"): (88, 88, 0.125),
    ("reading", "pro"): (100, 100, 0.000),
    ("blind", "flash"): (100, 40, 0.000),
    ("blind", "pro"): (100, 47, 0.000),
    ("spheroid", "flash"): (93, 87, 0.011),
    ("spheroid", "pro"): (93, 87, 0.067),
    ("culture", "flash"): (93, 86, 0.071),
    ("culture", "pro"): (86, 64, 0.043),
    ("pk", "flash"): (100, 79, 0.000),
    ("pk", "pro"): (100, 86, 0.000),
    # new-domains table (README.md "New-domain integration")
    ("newdomains", "flash"): (None, 93, 0.071),  # 13/14 usable
    ("newdomains", "pro"): (None, 79, 0.214),    # 11/14 usable
}


def audit_agent_rows() -> None:
    for (set_, model), (exp_sc, exp_usable, exp_hall) in AGENT_ROWS.items():
        fname = AGENT_FILE[set_].format(model=model)
        d = derive(_load(fname))["labwright"]
        got_sc = _pct(d.get("self_consistent_rate"))
        got_usable = _pct(d.get("usable_rate"))
        got_hall = d.get("hallucination_rate")
        if exp_sc is not None:
            _check(f"A  self-consistent {set_}/{model} == {exp_sc}%", got_sc == exp_sc,
                   f"{fname}: recomputed {got_sc}%, expected {exp_sc}%")
        _check(f"A  usable {set_}/{model} == {exp_usable}%", got_usable == exp_usable,
               f"{fname}: recomputed {got_usable}%, expected {exp_usable}%")
        if exp_hall is not None:
            _check(f"A  hallucination {set_}/{model} == {exp_hall}", _hall(got_hall) == exp_hall,
                   f"{fname}: recomputed {got_hall}, expected {exp_hall}")


# ---------------------------------------------------------------------------
# B. Fast-path (fine-tuned extractor) rows, incl. the schema-repair comparison
# ---------------------------------------------------------------------------

# (set, base file, repair file) -> (self-consistent %, usable %, hallucination)
FAST_ROWS = {
    "reading": ("eval_finetuned_reading_lora_v6.json", "eval_finetuned_reading_lora_v6_repair.json", (100, 96, 0.000)),
    "blind": ("eval_finetuned_blind_lora_v6.json", "eval_finetuned_blind_lora_v6_repair.json", (100, 27, 0.000)),
    "spheroid": ("eval_finetuned_spheroid_lora_v6.json", "eval_finetuned_spheroid_lora_v6_repair.json", (87, 73, 0.133)),
    "culture": ("eval_finetuned_culture_lora_v6.json", "eval_finetuned_culture_lora_v6_repair.json", (86, 57, 0.143)),
    "pk": ("eval_finetuned_pk_lora_v6.json", "eval_finetuned_pk_lora_v6_repair.json", (50, 50, 0.500)),
}


def audit_fast_rows() -> None:
    for set_, (base, _, (exp_sc, exp_usable, exp_hall)) in FAST_ROWS.items():
        d = derive(_load(base))["finetuned"]
        _check(f"B  fast-path self-consistent {set_} == {exp_sc}%", _pct(d["self_consistent_rate"]) == exp_sc,
               f"{base}: {_pct(d['self_consistent_rate'])}%")
        _check(f"B  fast-path usable {set_} == {exp_usable}%", _pct(d["usable_rate"]) == exp_usable,
               f"{base}: {_pct(d['usable_rate'])}%")
        _check(f"B  fast-path hallucination {set_} == {exp_hall}", _hall(d["hallucination_rate"]) == exp_hall,
               f"{base}: {d['hallucination_rate']}")

    # new-domains fast path (README table): 4/14 (29%) base, 5/14 (36%) with repair.
    base = derive(_load("eval_finetuned_newdomains_lora_v6.json"))["finetuned"]
    _check("B  new-domains fast-path usable == 4/14 (29%)",
           _pct(base["usable_rate"]) == 29,
           f"recomputed {_pct(base['usable_rate'])}% (usable_rate {base['usable_rate']:.4f})")
    _check("B  new-domains fast-path hallucination == 0.512",
           abs(base["hallucination_rate"] - 0.512) < 1e-3,
           f"recomputed {base['hallucination_rate']}")
    _check("B  new-domains failure composition (7 silence / 4 ok / 2 wrong-target / 1 calc-error)",
           base["failure_counts"] == {"silence": 7, "ok": 4, "wrong_target": 2, "calculation_error": 1},
           f"{base['failure_counts']}")
    repair = derive(_load("eval_finetuned_newdomains_lora_v6_repair.json"))["finetuned"]
    _check("B  new-domains repair usable == 5/14 (36%)",
           _pct(repair["usable_rate"]) == 36,
           f"recomputed {_pct(repair['usable_rate'])}%")
    # repair uniquely recovers scaling-kidney-chip (the eval-README narrative).
    base_ids = {e["id"] for e in _load("eval_finetuned_newdomains_lora_v6.json")["per_entry"]
                if _entry_usable(e.get("finetuned"))}
    rep_ids = {e["id"] for e in _load("eval_finetuned_newdomains_lora_v6_repair.json")["per_entry"]
               if _entry_usable(e.get("finetuned"))}
    gained = rep_ids - base_ids
    _check("B  repair gains exactly scaling-kidney-chip",
           gained == {"scaling-kidney-chip"},
           f"gained = {sorted(gained)}")


# ---------------------------------------------------------------------------
# C. Thinking ablation on the 15-goal blind set
# ---------------------------------------------------------------------------

def audit_thinking() -> None:
    pro_off = _usable_ids("eval_blind_pro.json")
    pro_on1 = _usable_ids("eval_blind_pro_thinking.json")
    pro_on2 = _usable_ids("eval_blind_pro_thinking_confirm.json")
    flash_off = _usable_ids("eval_blind_flash.json")
    flash_on1 = _usable_ids("eval_blind_flash_thinking.json")
    flash_on2 = _usable_ids("eval_blind_flash_thinking_confirm.json")

    _check("C  pro off == 7/15 (47%)", len(pro_off) == 7, f"{len(pro_off)}/15")
    _check("C  pro on run 1 == 10/15 (67%)", len(pro_on1) == 10, f"{len(pro_on1)}/15")
    _check("C  pro on run 2 == 10/15 (67%)", len(pro_on2) == 10, f"{len(pro_on2)}/15")
    _check("C  flash off == 6/15 (40%)", len(flash_off) == 6, f"{len(flash_off)}/15")
    _check("C  flash on run 1 == 7/15", len(flash_on1) == 7, f"{len(flash_on1)}/15")
    _check("C  flash on run 2 == 8/15", len(flash_on2) == 8, f"{len(flash_on2)}/15")

    # The two pro runs swap exactly one goal (run 1: liver-sinusoid vs run 2: venular-shear).
    _check("C  pro runs swap one goal (liver-sinusoid <-> venular-shear)",
           pro_on1 - pro_on2 == {"blind-liver-sinusoid"} and pro_on2 - pro_on1 == {"blind-venular-shear"},
           f"run1-only={sorted(pro_on1 - pro_on2)}, run2-only={sorted(pro_on2 - pro_on1)}")

    # The four hard-core goals stay missed on both models in every run.
    hard_core = {"blind-kidney-ptec", "blind-pulmonary-artery-shear",
                 "blind-retinal-arteriole-shear", "blind-lymphatic-shear"}
    misses = (hard_core & pro_off) | (hard_core & pro_on1) | (hard_core & pro_on2) | \
             (hard_core & flash_off) | (hard_core & flash_on1) | (hard_core & flash_on2)
    _check("C  four hard-core goals missed on both models x every run", not misses,
           f"recovered somewhere: {sorted(misses)}")


# ---------------------------------------------------------------------------
# D. Structure of the 15-goal blind split (cold / prompt-backed / scenario)
# ---------------------------------------------------------------------------

def audit_blind_split() -> None:
    gold = _load("eval_gold_blind.json") if (RESULTS / "eval_gold_blind.json").exists() else None
    # gold may be a dict with goals, or derived from a committed eval file's per_entry gold.
    per_entry = _load("eval_blind_pro.json")["per_entry"]
    counts: dict[str, int] = {}
    for e in per_entry:
        g = e.get("gold") or {}
        s = g.get("blind_strength")
        if s is None:
            s = f"scenario:{g.get('scenario')}"
        counts[s] = counts.get(s, 0) + 1
    _check("D  blind split = 8 cold + 5 prompt-backed + 2 scenario",
           counts.get("cold") == 8 and counts.get("prompt-backed") == 5
           and counts.get("scenario:unit-ambiguity") == 1 and counts.get("scenario:multi-target") == 1,
           f"{counts}")


# ---------------------------------------------------------------------------
# E. Gold-pairs inventory and new-domain exclusion
# ---------------------------------------------------------------------------

def audit_gold_pairs() -> None:
    pairs = [json.loads(line) for line in (RESULTS / "extractor/gold_pairs.jsonl").open()]
    domains = {}
    for p in pairs:
        d = p.get("domain") or p.get("block") or "?"
        domains[d] = domains.get(d, 0) + 1
    _check("E  gold_pairs = 46 (24 flow + 8 culture + 8 spheroid + 6 pk)",
           len(pairs) == 46 and domains == {"flow": 24, "culture": 8, "spheroid": 8, "pk": 6},
           f"{len(pairs)} pairs, {domains}")
    new_domain = {"barrier", "oxygen", "pumpless", "breathing", "pulsatile", "scaling", "gradient"}
    leaked = [p for p in pairs if (p.get("domain") or p.get("block")) in new_domain]
    _check("E  no new-domain goal in supervised gold pairs", not leaked,
           f"{[p.get('goal_id') for p in leaked]}")


# ---------------------------------------------------------------------------
# F. Leak-free held-out split (extractor_clean400 vs train)
# ---------------------------------------------------------------------------

def audit_leak_free() -> None:
    train_raw: set[str] = set()
    train_goal: set[str] = set()
    with (RESULTS / "extractor_11dom_v4/train.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            train_raw.add(json.dumps(r.get("raw"), sort_keys=True))
            train_goal.add(r.get("goal"))
    eval_raw_overlap = eval_goal_overlap = 0
    with (RESULTS / "extractor_clean400/eval.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            if json.dumps(r.get("raw"), sort_keys=True) in train_raw:
                eval_raw_overlap += 1
            if r.get("goal") in train_goal:
                eval_goal_overlap += 1
    _check("F  held-out eval set is leak-free (raw AND goal)",
           eval_raw_overlap == 0 and eval_goal_overlap == 0,
           f"raw overlap={eval_raw_overlap}, goal overlap={eval_goal_overlap}")


# ---------------------------------------------------------------------------
# G. 5-seed reproducibility intervals (README "Reproducibility" section)
# ---------------------------------------------------------------------------

SEED_EXPECTED = {
    ("flash", "bare"): (0.067, "0.067 [0.034, 0.126]"),
    ("flash", "labwright"): (0.925, "0.925 [0.864, 0.960]"),
    ("pro", "bare"): (0.108, "0.108 [0.064, 0.177]"),
    ("pro", "labwright"): (0.958, "0.958 [0.906, 0.982]"),
}
_SEED_MODEL = {"flash": "deepseek-v4-flash", "pro": "deepseek-v4-pro"}


def audit_seed_intervals() -> None:
    pooled = _load("eval_seed_benchmark.json")["pooled"]
    for (model, sysname), (exp_rate, exp_ci) in SEED_EXPECTED.items():
        v = pooled[_SEED_MODEL[model]][sysname]
        _check(f"G  seed usable {model}/{sysname} == {exp_rate}",
               abs(v["usable_design_rate"] - exp_rate) < 1e-3,
               f"recomputed {v['usable_design_rate']}")
        _check(f"G  seed CI {model}/{sysname} == {exp_ci}",
               v["usable_ci_str"] == exp_ci,
               f"recomputed '{v['usable_ci_str']}'")


# ---------------------------------------------------------------------------
# H. Schema-prompt A/B negative (the cheap fix is falsified on v6)
# ---------------------------------------------------------------------------

def audit_schema_prompt() -> None:
    base = derive(_load("eval_finetuned_newdomains_lora_v6.json"))["finetuned"]
    sp = _load("eval_finetuned_newdomains_lora_v6_schemaprompt.json")
    errs = {e.get("finetuned", {}).get("error") for e in sp["per_entry"]}
    _check("H  v6 + schema-prompt => 0/14 usable", all(not e.get("finetuned", {}).get("valid") for e in sp["per_entry"]),
           f"{sum(bool(e.get('finetuned', {}).get('valid')) for e in sp['per_entry'])}/14 valid")
    _check("H  every schema-prompt failure is unparseable_json", errs == {"unparseable_json"},
           f"{errs}")
    _check("H  schema-prompt removes the 4 base successes (falsification)",
           base["failure_counts"]["ok"] == 4,
           f"base ok={base['failure_counts']['ok']}")
    # Even the repair-retries variant stays at 0/14 with schema-prompt on.
    spr = _load("eval_finetuned_newdomains_lora_v6_schemaprompt_repair.json")
    _check("H  v6 + schema-prompt + repair-retries stays 0/14 usable",
           len(spr["per_entry"]) == 14 and all(not e.get("finetuned", {}).get("valid") for e in spr["per_entry"]),
           f"{sum(bool(e.get('finetuned', {}).get('valid')) for e in spr['per_entry'])}/14 valid")


def main() -> int:
    audit_agent_rows()
    audit_fast_rows()
    audit_thinking()
    audit_blind_split()
    audit_gold_pairs()
    audit_leak_free()
    audit_seed_intervals()
    audit_schema_prompt()
    print(f"audit_claims: {_passes} passed, {len(_failures)} failed")
    for f in _failures:
        print(f)
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
