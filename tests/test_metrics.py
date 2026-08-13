"""Tests for the blind-set metrics added in the metrics expansion.

Covers the failure-reason taxonomy (:func:`classify_failure`), the unit-misread
classifier wiring (:func:`unit_misreads`), and the three new blind scenarios —
unit-ambiguity, partial-info, multi-target — including that the gold entries
are *actually satisfiable* (a self-check that prevents an unwinnable gold from
silently inflating failure rates).
"""

import os

import pytest

from labwright.calc import culture as calc_culture
from labwright.calc import microfluidics as mf
from eval.benchmark import GoldExperiment, classify_failure, load_gold, unit_misreads

BLIND_GOLD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval", "gold_blind.json")


def _gold(gold_path=BLIND_GOLD):
    return load_gold(gold_path)


def test_gold_entries_parse_with_scenarios():
    entries = _gold()
    by_id = {g.id: g for g in entries}
    assert by_id["blind-kidney-ptec-unit-ambiguity"].scenario == "unit-ambiguity"
    assert by_id["blind-kidney-ptec-unit-ambiguity"].blind_strength is None
    assert by_id["blind-24well-medium-partial"].scenario == "partial-info"
    assert by_id["blind-24well-medium-partial"].blind_strength == "cold"
    assert by_id["blind-bbb-shear-residence-multitarget"].scenario == "multi-target"
    assert by_id["blind-bbb-shear-residence-multitarget"].blind_strength is None


def test_partial_info_gold_is_satisfiable():
    # 24-well: 6 wells × 0.68 mL/well = 4.08 mL.
    g = next(x for x in _gold() if x.id == "blind-24well-medium-partial")
    med = calc_culture.medium_volume_per_well("24")
    assert med == pytest.approx(0.68)
    assert 6 * med == pytest.approx(g.expected["total_medium_ml"])


def test_multi_target_gold_is_jointly_satisfiable():
    # In a 400×100 µm × 100 mm channel, shear 1.0 Pa needs Q ≈ 40 µL/min,
    # which gives residence time V/Q = 6.0 s — both targets at once.
    g = next(x for x in _gold() if x.id == "blind-bbb-shear-residence-multitarget")
    q = mf.flow_rate_for_shear_stress(1.0, 400, 100, 1e-3)
    assert q == pytest.approx(40.0, rel=0.02)
    residence = mf.residence_time(q, 400, 100, 100)
    assert residence == pytest.approx(6.0, rel=0.05)
    assert residence >= g.expected["residence_time_s"]


def test_unit_ambiguity_gold_conversion():
    # 0.2 dyn/cm² = 0.02 Pa; the gold asks for Pa.
    g = next(x for x in _gold() if x.id == "blind-kidney-ptec-unit-ambiguity")
    assert g.expected["shear_pa"] == pytest.approx(0.02)
    # The dyn→Pa factor is 0.1.
    assert mf_convert_dyn_to_pa(0.2) == pytest.approx(0.02)


def mf_convert_dyn_to_pa(value_dyn):
    from labwright.verify.units import convert
    return convert(value_dyn, "dyn/cm**2", "Pa")


def test_classify_failure_labwright_ok():
    rec = {"plan": True, "hallucination_rate": 0.0, "valid": True}
    assert classify_failure(rec, _gold()[0]) == "ok"


def test_classify_failure_labwright_silence():
    rec = {"plan": False, "hallucination_rate": 1.0, "valid": False}
    assert classify_failure(rec, _gold()[0]) == "silence"


def test_classify_failure_labwright_calculation_error():
    rec = {"plan": True, "hallucination_rate": 0.5, "valid": False}
    assert classify_failure(rec, _gold()[0]) == "calculation_error"


def test_classify_failure_labwright_wrong_target():
    rec = {"plan": True, "hallucination_rate": 0.0, "valid": False}
    assert classify_failure(rec, _gold()[0]) == "wrong_target"


def test_classify_failure_memory_silence():
    rec = {"reported": {}, "hallucination_rate": 1.0, "valid": False}
    assert classify_failure(rec, _gold()[0]) == "silence"


def test_classify_failure_memory_wrong_target():
    rec = {"reported": {"shear_pa": 0.05}, "hallucination_rate": 0.0, "valid": False}
    assert classify_failure(rec, _gold()[0]) == "wrong_target"


def test_unit_misread_catches_kidney_claimed_as_pa():
    g = next(x for x in _gold() if x.id == "blind-kidney-ptec")
    # The model reports 0.2 "Pa" when the true value is 0.02 Pa.
    misreads = unit_misreads({"shear_pa": 0.2}, g)
    assert "shear_pa" in misreads
    assert misreads["shear_pa"]["alias"] == "dyn/cm^2 read as Pa"


def test_unit_misread_correct_value_is_clean():
    g = next(x for x in _gold() if x.id == "blind-kidney-ptec")
    assert unit_misreads({"shear_pa": 0.02}, g) == {}


def test_unit_misread_on_unit_ambiguity_gold():
    g = next(x for x in _gold() if x.id == "blind-kidney-ptec-unit-ambiguity")
    # A dyn/cm²-as-Pa misread on the goal that states dyn/cm² → 0.2 vs 0.02.
    misreads = unit_misreads({"shear_pa": 0.2}, g)
    assert "shear_pa" in misreads


def test_no_unit_misread_on_wrong_magnitude():
    g = next(x for x in _gold() if x.id == "blind-arterial-shear")
    # 3× the target is a plain arithmetic error, not a known alias pair.
    assert unit_misreads({"shear_pa": 4.5}, g) == {}
