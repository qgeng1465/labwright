"""Tests for the allometric / physiological scaling calculators.

The equations must reproduce the body-on-chip design rules: cardiac-output
fractions per organ, mass-proportional cell budgets, Kleiber metabolic
allometry and transit-time matching.
"""

import pytest

from labwright.calc import scaling as s


# -- Flow scaling -------------------------------------------------------------


def test_organ_flow_fraction_known_values():
    assert s.organ_flow_fraction("liver") == pytest.approx(0.27, rel=1e-9)
    assert s.organ_flow_fraction("kidneys") == pytest.approx(0.22, rel=1e-9)
    assert s.organ_flow_fraction("heart") == pytest.approx(0.05, rel=1e-9)


def test_organ_flow_rate_known_value():
    # Liver gets 27% of a 5 L/min cardiac output → ≈ 1350 mL/min.
    assert s.organ_flow_rate_mlmin("liver", 5000) == pytest.approx(1350.0, rel=1e-9)


def test_organ_flow_rate_scales_with_cardiac_output():
    assert s.organ_flow_rate_mlmin("liver", 500) == pytest.approx(135.0, rel=1e-9)


def test_unknown_organ_raises():
    with pytest.raises(ValueError):
        s.organ_flow_fraction("spleen")


# -- Cell and metabolic scaling -----------------------------------------------


def test_scale_cell_number_known_value():
    # 1.5 kg liver in a 70 kg body with a 10⁶-cell budget → ≈ 21429 cells.
    assert s.scale_cell_number(1500, 70000, 1e6) == pytest.approx(21428.57, rel=1e-4)


def test_scale_cell_number_preserves_mass_ratio():
    # Twice the organ mass → twice the cells.
    n1 = s.scale_cell_number(1500, 70000, 1e6)
    n2 = s.scale_cell_number(3000, 70000, 1e6)
    assert n2 == pytest.approx(2.0 * n1, rel=1e-9)


def test_allometric_metabolic_scale_known_value():
    # (1500/70000)^0.75 ≈ 0.056 — Kleiber exponent < 1 damps the mass fraction.
    assert s.allometric_metabolic_scale(1500, 70000, 0.75) == pytest.approx(0.0562, rel=1e-2)


def test_allometric_mass_proportional_exponent():
    # exponent 1 → exactly the mass fraction.
    assert s.allometric_metabolic_scale(1500, 70000, 1.0) == pytest.approx(1500 / 70000, rel=1e-9)


def test_allometric_damps_the_mass_fraction():
    # Kleiber: for organ mass < body mass, (m/M)^0.75 > (m/M)^1 — the organ's
    # metabolic share exceeds its raw mass fraction. A 1.5 kg liver behaves as
    # ~5.6% of body metabolism, not 2.1%.
    assert s.allometric_metabolic_scale(1500, 70000, 0.75) > s.allometric_metabolic_scale(1500, 70000, 1.0)


# -- Transit / residence time -------------------------------------------------


def test_transit_time_known_value():
    # 1 mL at 100 µL/min → 600 s.
    assert s.transit_time_s(1000, 100) == pytest.approx(600.0, rel=1e-9)


def test_residence_time_match_zero_at_target():
    assert s.residence_time_match_error_s(1000, 100, 600) == pytest.approx(0.0, rel=1e-9)


def test_residence_time_match_reports_residual():
    assert s.residence_time_match_error_s(1000, 100, 900) == pytest.approx(300.0, rel=1e-9)


def test_transit_time_validates():
    with pytest.raises(ValueError):
        s.transit_time_s(1000, 0)


# -- Chip design --------------------------------------------------------------


def test_chip_scale_factor():
    out = s.chip_scale_factor_for_organ(1500, 70000, 1e6)
    assert out["scale"] == pytest.approx(0.02142857, rel=1e-5)
    assert out["cells_in_chip"] == pytest.approx(21428.57, rel=1e-4)
