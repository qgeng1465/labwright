"""Unit tests for labwright.calc.dosing."""

import pytest

from labwright.calc import dosing


def test_molarity_roundtrip():
    # 5 mg of a 500 g/mol compound in 1 mL -> 5/500/1 = 0.01 mM
    assert dosing.molarity_from_mass(5, 500, 1) == pytest.approx(0.01)
    # inverse
    assert dosing.mass_for_molarity(0.01, 500, 1) == pytest.approx(5)


def test_dilution():
    # 100 mM stock -> 1 mM working in 10 mL: add 0.1 mL
    assert dosing.dilution_volume(100, 1, 10) == pytest.approx(0.1)


def test_dilution_needs_diluting_stock():
    with pytest.raises(ValueError):
        dosing.dilution_volume(1, 100, 10)


def test_final_concentration():
    # 0.1 mL of 100 mM into 10 mL total -> 1 mM
    assert dosing.final_concentration_after_dilution(100, 0.1, 10) == pytest.approx(1)


def test_serial_dilution():
    concs = dosing.serial_dilution(100, 3, 3)
    assert concs == pytest.approx([100, 100 / 3, 100 / 9, 100 / 27])


def test_serial_dilution_steps_must_be_positive_int():
    with pytest.raises(ValueError):
        dosing.serial_dilution(100, 3, 0)


def test_dmso_fraction():
    # 50 mM DMSO stock, 0.1 mM working -> 0.2% v/v
    assert dosing.dmso_fraction(50, 0.1) == pytest.approx(0.002)


def test_dmso_above_stock_raises():
    with pytest.raises(ValueError):
        dosing.dmso_fraction(0.1, 50)


def test_molar_to_mass_conc():
    # 1 uM of 400 g/mol = 0.4 ng/mL
    assert dosing.molar_to_ng_per_ml(0.001, 400) == pytest.approx(400)
