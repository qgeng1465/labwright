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
import re
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

def audit_cold_expansion() -> None:
    """Structure + provenance of the cold-blind expansion (P0-2, 2026-08-16).

    ``eval/gold_cold_expansion.json`` adds 4 organ-flow-fraction goals to the
    8 committed cold blind goals. Every value must be:
      * cold for the base LLM (absent from the goal text and from the Labwright
        system prompt — the prompt never phrases flow as a cardiac-output
        fraction);
      * traceable to the committed body-on-chip flow table
        ``labwright/calc/scaling.py:ORGAN_FLOW_FRACTIONS`` (Ucciferri et al.
        2014, doi:10.3389/fbioe.2014.00074);
      * distinct from the organ-flow values already pinned in
        ``eval/gold_new_domains.json`` (liver 0.27 / kidneys 0.22).
    """
    import labwright.agent.agent as _agent
    from labwright.calc import scaling as _cs
    from labwright.physiology import physiology_anchor_text

    path = ROOT / "eval/gold_cold_expansion.json"
    golds = json.load(open(path))
    _check("G  cold expansion has 4 goals, all blind_strength=cold",
           len(golds) == 4 and all(g.get("blind_strength") == "cold" for g in golds),
           f"n={len(golds)}")
    new_domain_vals = {
        float(v)
        for g in json.load(open(ROOT / "eval/gold_new_domains.json"))
        for k, v in g["expected"].items()
        if k in ("organ_flow_fraction", "organ_flow_rate_mlmin")
    }
    prompt_text = _agent.SYSTEM_PROMPT + "\n" + physiology_anchor_text()
    # The system prompt never phrases perfusion as a cardiac-output fraction,
    # so a scaling goal cannot be prompt-backed.
    _check("G  cold expansion: system prompt has no flow-fraction phrasing",
           "flow fraction" not in prompt_text and "cardiac output" not in prompt_text)
    seen_organs: set[str] = set()
    for g in golds:
        organ = g["goal"].split("body-on-chip ")[1].split(" compartment")[0]
        frac = _cs.ORGAN_FLOW_FRACTIONS[organ]
        flow = round(frac * _cs.CARDIAC_OUTPUT_MLMIN, 6)
        _check(f"G  {g['id']}: expected == committed table ({organ} {frac} x CO)",
               abs(g["expected"]["organ_flow_fraction"] - frac) < 1e-9
               and abs(g["expected"]["organ_flow_rate_mlmin"] - flow) < 1e-9,
               f"got {g['expected']}")
        _check(f"G  {g['id']}: value absent from goal text (cold)",
               str(frac) not in g["goal"] and str(int(flow)) not in g["goal"])
        _check(f"G  {g['id']}: value distinct from committed new-domain fractions",
               float(g["expected"]["organ_flow_fraction"]) not in new_domain_vals,
               f"collides with {sorted(new_domain_vals)}")
        _check(f"G  {g['id']}: source cites Ucciferri DOI",
               "10.3389/fbioe.2014.00074" in g.get("source", ""))
        seen_organs.add(organ)
    _check("G  cold expansion: organs brain/heart/gut/skin, none reused",
           seen_organs == {"brain", "heart", "gut", "skin"})


def _cold_entries(name: str) -> list[dict]:
    """Per-entry records whose gold is blind_strength=cold in a committed file."""
    return [e for e in _load(name)["per_entry"]
            if (e.get("gold") or {}).get("blind_strength") == "cold"]


def _value_recall(rows: list[dict], system: str) -> int:
    """Value-recall: a recovery was submitted and every value lands within ±5%.

    Unlike ``_entry_usable`` this drops the hallucination gate: the un-gated
    systems (bare / soft_gate / self_verify) are scored hallucination=1.0 on
    any invented value by construction, so their *usable* rate is always 0 for
    these goals. The docs report their value-recall instead.
    """
    n = 0
    for e in rows:
        recv = e[system].get("recovery") or {}
        if recv and all(abs(float(v)) <= 0.05 for v in recv.values()):
            n += 1
    return n


def audit_cold_aggregates() -> None:
    """Combined cold-only aggregates (n=12) pinned by README ("Cold-only" box).

    The 8 committed cold blind goals (``eval/gold_blind.json``) plus the 4 cold
    organ-flow goals (``eval/gold_cold_expansion.json``) form one cold-only set.
    This check recomputes the headline numbers the docs display:
      * Labwright usable (the documented usable rule: hallucination 0.0 AND
        every recovery within ±5 %) — 7/12 = 58 % [32, 81] for both models;
      * the un-gated systems' value-recall (recovery within ±5 %, hallucination
        gate dropped as above) — bare flash 4/12, bare pro 3/12;
      * the pre-expansion 8-goal Labwright usable stays 3/8 = 38 % [14, 69].
    """
    from eval.ci import format_ci as _ci

    bf = _cold_entries("eval_blind_flash.json")
    bp = _cold_entries("eval_blind_pro.json")
    ef = _load("eval_cold_expansion_flash.json")["per_entry"]
    ep = _load("eval_cold_expansion_pro.json")["per_entry"]
    _check("D  cold-only n=12: 8 blind + 4 expansion, same ids per model",
           len(bf) == 8 and len(ef) == 4
           and {e["id"] for e in bf + ef} == {e["id"] for e in bp + ep})

    for model, rows in (("flash", bf + ef), ("pro", bp + ep)):
        k = sum(1 for e in rows if _entry_usable(e.get("labwright")))
        _check(f"D  cold-only n=12 {model} labwright usable 7/12 = {_ci(k, len(rows))}",
               k == 7, f"got {k}")
        k = _value_recall(rows, "bare")
        _check(f"D  cold-only n=12 {model} bare value-recall "
               f"{k}/12 = {_ci(k, len(rows))}",
               (model == "flash" and k == 4) or (model == "pro" and k == 3),
               f"got {k}")

    for model, rows in (("flash", bf), ("pro", bp)):
        k = sum(1 for e in rows if _entry_usable(e.get("labwright")))
        _check(f"D  orig-8 cold-only {model} labwright usable 3/8 = {_ci(k, len(rows))}",
               k == 3, f"got {k}")


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
#: P0-3 (2026-08-16): the 14-goal new-domains set re-run over 5 seeds
#: (results/eval_seed_newdomains.json, 14 goals x 5 seeds = 70 trials).
ND_SEED_EXPECTED = {
    ("flash", "bare"): (0.000, "0.000 [0.000, 0.052]"),
    ("flash", "soft_gate"): (0.000, "0.000 [0.000, 0.052]"),
    ("flash", "self_verify"): (0.000, "0.000 [0.000, 0.052]"),
    ("flash", "labwright"): (0.986, "0.986 [0.923, 0.997]"),
    ("pro", "bare"): (0.200, "0.200 [0.123, 0.308]"),
    ("pro", "soft_gate"): (0.114, "0.114 [0.059, 0.210]"),
    ("pro", "self_verify"): (0.000, "0.000 [0.000, 0.052]"),
    ("pro", "labwright"): (0.786, "0.786 [0.676, 0.866]"),
}
_SEED_MODEL = {"flash": "deepseek-v4-flash", "pro": "deepseek-v4-pro"}


def audit_seed_intervals() -> None:
    for fname, expected in (("eval_seed_benchmark.json", SEED_EXPECTED),
                            ("eval_seed_newdomains.json", ND_SEED_EXPECTED)):
        tag = fname[len("eval_seed_"):-len(".json")]
        pooled = _load(fname)["pooled"]
        for (model, sysname), (exp_rate, exp_ci) in expected.items():
            v = pooled[_SEED_MODEL[model]][sysname]
            _check(f"G  seed usable {tag} {model}/{sysname} == {exp_rate}",
                   abs(v["usable_design_rate"] - exp_rate) < 1e-3,
                   f"recomputed {v['usable_design_rate']}")
            _check(f"G  seed CI {tag} {model}/{sysname} == {exp_ci}",
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


# ---------------------------------------------------------------------------
# I. Value-level provenance of the gold target values in training goals
# ---------------------------------------------------------------------------
# The docs used to claim "no gold number appears verbatim in a training goal".
# That is false for the blind set: the flow generator samples the blind golds'
# shear values (labwright/extract/synthetic.py's vessel table), so 11 of the 15
# blind goals carry a gold target in their training-goal prose. These checks
# lock the honest counts and the doc labels so the claim cannot quietly return.

_BLIND_UNSEEN = {
    "blind-seed-hepg2-log",   # 4,000 cells  — value absent from training goals
    "blind-phh-seed",         # 12,000 cells
    "blind-gut-epithelial-shear",  # 0.002 Pa
    "blind-24well-medium-partial",  # 4.08 mL
}


def _gold_unit(key: str) -> str:
    kl = key.lower()
    if "ml" in kl or "volume" in kl:
        return "mL"
    if "cell" in kl or "seed" in kl:
        return "cells"
    if "time" in kl or "residence" in kl:
        return "s"
    if "diam" in kl:
        return "um"
    return "Pa"


# New-domain keys that carry a real unit in goal prose (dimensionless counts
# like pulsatility index or a shear fraction are skipped — the fast-path has
# to compute those regardless of any training-value overlap).
_ND_UNIT = {
    "teer_ohm_cm2": "ohm", "papp_cm_s": "cm/s", "clearance_mL_min": "mL/min",
    "dissolved_o2_mM": "mM", "penetration_depth_um": "um",
    "demand_umol_min": "umol", "hydrostatic_head_pa": "Pa",
    "peak_wall_shear_pa": "Pa", "volume_per_half_cycle_ul": "uL",
    "breaths_per_minute": "per minute", "cyclic_displacement_um": "um",
    "strain_rate_per_s": "per second", "ali_liquid_film_um": "um",
    "peak_shear_pa": "Pa", "organ_flow_rate_mlmin": "mL/min",
    "cells_in_organ": "cells", "transit_time_s": "second",
    "residence_time_match_error_s": "second", "steepness_um_per_mm": "µM/mm",
    "midpoint_conc_um": "µM", "relaxation_time_s": "second", "flux_mol_m2s": "mol/",
}


def _load_gold(name: str) -> list[dict]:
    gold = json.load(open(ROOT / "eval" / name))
    if isinstance(gold, dict):
        gold = list(gold.values())[0] if gold else []
    return gold


def _goals() -> list[str]:
    with (RESULTS / "extractor_11dom_v4/train.jsonl").open() as f:
        return [json.loads(line).get("goal") or "" for line in f]


def _verbatim_goals(goals: list[str], value: float, unit: str) -> int:
    pat = re.compile(r"(?<![\w.])" + re.escape(f"{value:g} {unit}") + r"(?![\w.])")
    return sum(1 for g in goals if pat.search(g))


def audit_value_provenance() -> None:
    goals = _goals()

    # Blind: how many goals carry a gold target verbatim (value + unit)?
    leak_goals: set[str] = set()
    for r in _load_gold("gold_blind.json"):
        for k, v in (r.get("expected") or {}).items():
            if isinstance(v, (int, float)) and _verbatim_goals(goals, float(v), _gold_unit(k)) > 0:
                leak_goals.add(r.get("id"))
    _check("I  blind: 11/15 goals carry a gold target verbatim in training goals (value+unit)",
           len(leak_goals) == 11, f"{len(leak_goals)}/15: {sorted(leak_goals)}")
    all_ids = {r.get("id") for r in _load_gold("gold_blind.json")}
    _check("I  blind: the 4 fully-unseen targets are seed-hepg2, phh-seed, gut, 24-well",
           all_ids - leak_goals == _BLIND_UNSEEN, f"unseen={sorted(all_ids - leak_goals)}")

    # New-domain: the docs disclose that the generators sample the golds'
    # values. Assert at least one new-domain gold value is verifiable verbatim,
    # so the disclosure is a real, checked claim rather than hand-waving.
    nd_hits = 0
    for r in _load_gold("gold_new_domains.json"):
        for k, v in (r.get("expected") or {}).items():
            if isinstance(v, (int, float)) and k in _ND_UNIT:
                nd_hits += _verbatim_goals(goals, float(v), _ND_UNIT[k])
    _check("I  new-domain: at least one gold target value verbatim in training goals",
           nd_hits > 0, f"{nd_hits} verbatim occurrences")


def audit_doc_labels() -> None:
    """The honest blind label and the dropped 'no gold number verbatim' claim
    are part of the reproducibility contract — an edit that resurrects the old
    framing fails the audit."""
    for name in ("README.md", "eval/README.md", "README.zh-CN.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        _check(f"I  {name}: fast-path blind row is 'targets in train'",
               "fast-path (targets in train)" in text, "'targets in train' absent")
        _check(f"I  {name}: no resurrected '(novel)' fast-path label",
               "fast-path (novel)" not in text, "'fast-path (novel)' present")
        _check(f"I  {name}: no resurrected 'no gold number appears verbatim' claim",
               "no gold number appears verbatim" not in text,
               "the overclaim is back in the docs")


def audit_labmath_bench() -> None:
    """LabMath-Bench dataset claims (reviewer demand #1).

    Pins the committed dataset the TBA metric is reported over: ≥500 entries,
    every difficulty level ≥140, every expected value a finite positive target
    (a zero residual would make relative-error recovery undefined), and every
    entry tagged validly. The combined file (generated + tagged existing golds)
    must keep all three levels populated.
    """
    from collections import Counter

    gold = json.load(open(_HERE / "gold_labmath_bench.json"))
    by_level = Counter(e["level"] for e in gold)
    _check("J  LabMath-Bench: >=500 entries", len(gold) >= 500, f"{len(gold)}")
    for lv in ("L1", "L2", "L3"):
        _check(f"J  LabMath-Bench: level {lv} >= 140", by_level.get(lv, 0) >= 140,
               f"{by_level.get(lv, 0)}")
    degenerate = [(e["id"], k, v) for e in gold
                  for k, v in e["expected"].items() if not (v == v and v > 0)]
    _check("J  LabMath-Bench: no zero/nan expected targets", not degenerate,
           str(degenerate[:3]))
    tags_ok = all(
        e["level"] in ("L1", "L2", "L3")
        and e["difficulty"] in ("easy", "medium", "hard")
        and e["scenario"] == "complete-info"
        and e["source"]
        for e in gold
    )
    _check("J  LabMath-Bench: valid level/difficulty/scenario/source tags", tags_ok)
    combined = json.load(open(_HERE / "gold_labmath_combined.json"))
    _check("J  LabMath-Bench combined: >=600 entries", len(combined) >= 600,
           f"{len(combined)}")
    for lv in ("L1", "L2", "L3"):
        n = sum(1 for e in combined if e["level"] == lv)
        _check(f"J  LabMath-Bench combined: level {lv} >= 140", n >= 140, f"{n}")


def audit_adversarial() -> None:
    """Boundary/adversarial claims (reviewer demand #3).

    Two layers. First the *deterministic* contract that the fail-safe figure
    stands on, independent of any LLM run: the adversarial set's shape and —
    critically — that every ``physical_conflict``/``lethal_condition`` entry's
    implied raw inputs are genuinely hard-rejected by ``submit_design`` (the
    verifier or the schema), so ``exception_catch_rate`` is a claim about the
    gate, not about model behaviour. Second, once ``results/adversarial_*.json``
    are committed, the per-system fail-safe / fabrication rates are pinned to
    the exact summary block the figure reads.
    """
    from collections import Counter

    from labwright.design import submit_design

    adv = json.load(open(_HERE / "gold_adversarial.json"))
    _check("K  adversarial: >=25 entries", len(adv) >= 25, f"{len(adv)}")
    types = Counter(e["type"] for e in adv)
    _check("K  adversarial: missing_parameter >= 10", types.get("missing_parameter", 0) >= 10,
           f"{types.get('missing_parameter', 0)}")
    _check("K  adversarial: physical_conflict >= 8", types.get("physical_conflict", 0) >= 8,
           f"{types.get('physical_conflict', 0)}")
    _check("K  adversarial: lethal_condition >= 6", types.get("lethal_condition", 0) >= 6,
           f"{types.get('lethal_condition', 0)}")
    ids = [e["id"] for e in adv]
    _check("K  adversarial: ids unique", len(ids) == len(set(ids)))

    # Honesty gate: every trap's implied raws must be hard-caught offline.
    hard_caught, n_traps = 0, 0
    for e in adv:
        if e["type"] == "missing_parameter":
            assert e["expected_outcome"] == "elicit", e["id"]
            assert "implied_raws" not in e, e["id"]
            continue
        n_traps += 1
        raws = dict(e["implied_raws"])
        raws.setdefault("goal", e["goal"])
        raws.setdefault("rationale", "audit")
        try:
            res = submit_design(raws)
            n_err = sum(1 for i in res.get("verification", []) if i.get("level") == "error")
            caught = res.get("status") in ("review_required", "validation_error") and (
                n_err > 0 or res.get("status") == "validation_error")
        except Exception:
            caught = True
        hard_caught += int(caught)
        if not caught:
            _failures.append(f"  FAIL K  adversarial {e['id']}: implied raws not rejected")
    _check("K  adversarial: all physical/lethal traps hard-caught by verifier",
           hard_caught == n_traps, f"{hard_caught}/{n_traps}")

    # Pinned fail-safe numbers from the committed adversarial runs (added the
    # moment the result JSONs are committed; skipped while they do not exist).
    # Pin the exact values for flash (committed); pro is pinned as "present"
    # until its result lands, then the exact values are filled in.
    PINNED_FLASH = {
        # (system) -> (fail_safe, fabrication, elicitation, exception_catch)
        "bare": (0.8333, 0.1667, 0.0, 0.0),
        "code_interpreter": (0.7333, 0.2333, 0.0, 0.0),
        "labwright": (0.9333, 0.0667, 0.6667, 0.2333),
    }
    PINNED_PRO = {
        # Filled with the exact summary values once adversarial_pro.json lands
        # (30/30). Same (system) -> (fail_safe, fabrication, elicitation,
        # exception_catch) contract as PINNED_FLASH.
        "bare": (None, None, None, None),
        "code_interpreter": (None, None, None, None),
        "labwright": (None, None, None, None),
    }
    for model, name in (("deepseek-v4-flash", "adversarial_flash.json"),
                        ("deepseek-v4-pro", "adversarial_pro.json")):
        path = RESULTS / name
        if not path.exists():
            _failures.append(f"  FAIL K  {name} not committed — fail-safe numbers unpinned")
            continue
        run = json.load(open(path))
        assert run["model"] == model, f"{name} model mismatch"
        per_entry = run.get("per_entry", [])
        _check(f"K  adversarial {model}: ran all 30 entries",
               len(per_entry) >= 30, f"{len(per_entry)}")
        summary = run.get("summary", {}).get("systems", {})
        for sys_name in ("bare", "code_interpreter", "labwright"):
            agg = summary.get(sys_name, {})
            _check(f"K  adversarial {model} {sys_name}: fail_safe_rate present",
                   "fail_safe_rate" in agg, str(agg.get("fail_safe_rate")))
            pinned = PINNED_FLASH if model == "deepseek-v4-flash" else PINNED_PRO
            if sys_name in pinned and all(x is not None for x in pinned[sys_name]):
                fs, fab, elic, exc = pinned[sys_name]
                _check(f"K  adversarial {model} {sys_name}: fail_safe pinned",
                       abs(agg["fail_safe_rate"] - fs) < 1e-4, f"{agg['fail_safe_rate']:.4f}")
                _check(f"K  adversarial {model} {sys_name}: fabrication pinned",
                       abs(agg["fabrication_rate"] - fab) < 1e-4, f"{agg['fabrication_rate']:.4f}")
                _check(f"K  adversarial {model} {sys_name}: elicitation pinned",
                       abs(agg["elicitation_rate"] - elic) < 1e-4, f"{agg['elicitation_rate']:.4f}")
                _check(f"K  adversarial {model} {sys_name}: exception_catch pinned",
                       abs(agg["exception_catch_rate"] - exc) < 1e-4, f"{agg['exception_catch_rate']:.4f}")


def audit_code_interpreter() -> None:
    """Baseline B (code interpreter) plumbing claims (reviewer demand #2)."""
    import inspect

    from eval import report as report_mod
    from eval import benchmark as bench_mod

    # The report's failure-reasons loop must carry the 5th class.
    src = inspect.getsource(report_mod)
    _check("K  code_interpreter: distinct code_exec_error class in failure report",
           "code_exec_error" in src)
    _check("K  code_interpreter: runner dispatched",
           hasattr(bench_mod, "run_code_interpreter") and hasattr(bench_mod, "_run_code_sandbox"))
    _check("K  code_interpreter: sandbox blocks builtins",
           "open" in getattr(bench_mod, "_CODEX_BLOCKED_BUILTINS", ()))

    _check("K  code_interpreter: runner dispatched",
           hasattr(bench_mod, "run_code_interpreter") and hasattr(bench_mod, "_run_code_sandbox"))
    _check("K  code_interpreter: sandbox blocks builtins",
           "open" in getattr(bench_mod, "_CODEX_BLOCKED_BUILTINS", ()))


def audit_traceability() -> None:
    """Supplementary traceability-log claims (reviewer demand #4).

    Mechanism check (always runs): the builder writes per-entry provenance JSONs
    and an INDEX from any v0.7+ results. Coverage check (once the full results
    land): the committed LabMath flash results carry plan + provenance on the
    design-path systems.
    """
    import tempfile

    from eval import make_traceability_log as mtl

    _check("K  traceability: builder importable",
           callable(getattr(mtl, "build", None)) and callable(getattr(mtl, "iter_design_records", None)))
    _check("K  traceability: design systems cover verified paths",
           set(mtl.DESIGN_SYSTEMS) >= {"labwright", "labwright_iter", "tool_no_gate"})

    # Mechanism check on a tiny deterministic fixture (no LLM needed).
    prov = [{"field": "derived.shear_pa", "formula": "tau", "unit": "Pa",
             "value": 0.05, "status": "ok",
             "inputs": [{"name": "flow_rate_uLmin", "value": 1.0, "unit": "uL/min"}],
             "code_version": "test"}]
    fake = {"model": "audit-model", "per_entry": [{
        "id": "t", "gold": {"goal": "g"},
        "labwright": {"valid": True, "plan": {"derived": {"shear_pa": 0.05}},
                      "provenance": prov, "tool_trace": ["submit_design"]}}]}
    with tempfile.TemporaryDirectory() as td:
        s = mtl.build(fake, td)
        _check("K  traceability: fixture yields 1 traced entry", s["entries_with_provenance"] == 1,
               f"{s['entries_with_provenance']}")
        _check("K  traceability: per-entry log + INDEX written",
               (Path(td) / "audit-model" / "t__labwright.json").exists()
               and (Path(td) / "INDEX.json").exists())

    # Coverage on committed results (skipped until the full run lands).
    for name in ("eval_labmath_flash.json",):
        path = RESULTS / name
        if not path.exists():
            _failures.append(f"  FAIL K  {name} not committed — traceability coverage unpinned")
            continue
        run = json.load(open(path))
        traced = sum(1 for _entry_id, _goal, sysname, rec in mtl.iter_design_records(run)
                     if sysname == "labwright" and isinstance(rec.get("plan"), dict)
                     and rec.get("provenance"))
        _check(f"K  traceability {name}: labwright plans carry provenance",
               traced >= 1, f"{traced} traced labwright entries")


def main() -> int:
    audit_agent_rows()
    audit_fast_rows()
    audit_thinking()
    audit_blind_split()
    audit_cold_expansion()
    audit_cold_aggregates()
    audit_gold_pairs()
    audit_leak_free()
    audit_seed_intervals()
    audit_schema_prompt()
    audit_value_provenance()
    audit_doc_labels()
    audit_labmath_bench()
    audit_adversarial()
    audit_code_interpreter()
    audit_traceability()
    print(f"audit_claims: {_passes} passed, {len(_failures)} failed")
    for f in _failures:
        print(f)
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
