"""Tests for the barrier-function calculators (TEER / Papp / flux / clearance).

The equations must reproduce the Transwell QC conventions, round-trip, and be
exposed as tools. The registry integration test pins that barrier lines carry a
TEER range so the agent can check a planned readout against the literature.
"""

import pytest

from labwright.calc import barrier as b
from labwright.physiology import lookup_cell
from labwright.tools import REGISTRY


# -- TEER ---------------------------------------------------------------------


def test_teer_from_resistance_known_value():
    # (300 − 100) Ω × 1.12 cm² = 224 Ω·cm².
    assert b.teer_ohm_cm2(300, 100, 1.12) == pytest.approx(224.0, rel=1e-9)


def test_teer_requires_subtracting_blank():
    # A larger blank must give a smaller TEER — skipping the blank overstates it.
    assert b.teer_ohm_cm2(300, 100, 1.12) < b.teer_ohm_cm2(300, 50, 1.12)


def test_transendothelial_resistance_inverse():
    assert b.transendothelial_resistance_ohm(224, 1.12) == pytest.approx(200.0, rel=1e-9)


def test_teer_roundtrip():
    teer = b.teer_ohm_cm2(500, 50, 0.33)
    assert b.transendothelial_resistance_ohm(teer, 0.33) == pytest.approx(450.0, rel=1e-9)


def test_teer_rejects_negative_residual():
    with pytest.raises(ValueError):
        b.teer_ohm_cm2(50, 300, 1.12)  # R_total < R_blank


# -- Permeability --------------------------------------------------------------


def test_papp_from_flux_known_value():
    # (0.006/60) nmol/s ÷ (1 cm² × 100 nmol/cm³) = 1e-6 cm/s.
    assert b.papp_cm_s(0.006, 1, 100) == pytest.approx(1e-6, rel=1e-9)


def test_flux_from_papp_inverse():
    # Papp 1e-6 cm/s, C₀ 100 µM, A 1.12 cm² → 6.72e-3 nmol/min (6.72 pmol/min).
    flux = b.flux_nmol_min(1e-6, 100, 1.12)
    assert flux == pytest.approx(6.72e-3, rel=1e-9)
    assert b.papp_cm_s(flux, 1.12, 100) == pytest.approx(1e-6, rel=1e-9)


def test_tight_barrier_flux_is_picomolar():
    # A 1e-7 cm/s barrier across a 0.33 cm² insert at 10 µM → pmol/min flux.
    flux = b.flux_nmol_min(1e-7, 10, 0.33)
    assert 1e-6 < flux < 1e-3  # sub-nmol/min — needs a sensitive readout


def test_clearance():
    # Papp 1e-6 × 1.12 cm² = 1.12e-6 cm³/s = 6.72e-5 mL/min.
    assert b.clearance_mL_min(1e-6, 1.12) == pytest.approx(6.72e-5, rel=1e-9)


def test_effective_permeability():
    # D = 1e-13 m²/s over 20 µm → 5e-9 m/s = 5e-7 cm/s.
    assert b.effective_permeability_cm_s(1e-13, 20) == pytest.approx(5e-7, rel=1e-9)


# -- Validation ---------------------------------------------------------------


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        b.teer_ohm_cm2(0, 100, 1.12)
    with pytest.raises(ValueError):
        b.papp_cm_s(1, 0, 100)
    with pytest.raises(ValueError):
        b.flux_nmol_min(1e-6, 0, 1)
    with pytest.raises(ValueError):
        b.effective_permeability_cm_s(0, 20)


# -- Tool registration --------------------------------------------------------


def test_barrier_tools_registered():
    for name in ("teer_from_resistance", "transendothelial_resistance",
                 "papp_from_flux", "flux_from_papp", "clearance_from_papp",
                 "effective_permeability"):
        assert name in REGISTRY, name


def test_teer_tool_matches_module():
    out = REGISTRY["teer_from_resistance"].func(300, 100, 1.12)
    assert out == pytest.approx(b.teer_ohm_cm2(300, 100, 1.12), rel=1e-9)


def test_papp_tool_matches_module():
    out = REGISTRY["papp_from_flux"].func(0.006, 1, 100)
    assert out == pytest.approx(1e-6, rel=1e-9)


# -- Registry integration -----------------------------------------------------


def test_barrier_lines_carry_teer_ranges():
    caco = lookup_cell("Caco-2")
    assert caco.barrier == "intestinal"
    assert caco.teer_ohm_cm2 == (250, 1000)
    bbb = lookup_cell("hCMEC/D3")
    assert bbb.barrier == "blood-brain"
    assert bbb.teer_ohm_cm2 == (100, 240)


def test_registry_distinguishes_model_from_physiological_teer():
    # hCMEC/D3's in-vitro range is ~10× below in-vivo BBB — the model limitation
    # must not be silently conflated with the physiological reference.
    bbb = lookup_cell("hCMEC/D3")
    assert bbb.teer_ohm_cm2[1] < bbb.teer_physiological_ohm_cm2[0]
