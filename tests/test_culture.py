"""Unit tests for labwright.calc.culture."""

import pytest

from labwright.calc import cell, culture


def test_plate_formats_complete():
    assert set(culture.PLATE_FORMATS) == {"6", "12", "24", "48", "96"}


def test_normalize_format():
    assert culture._normalize_format("96-well") == "96"
    assert culture._normalize_format("6well") == "6"
    assert culture._normalize_format("24") == "24"
    with pytest.raises(ValueError):
        culture._normalize_format("384")


def test_well_surface_area():
    assert culture.well_surface_area_cm2("96") == pytest.approx(0.32)
    assert culture.well_surface_area_cm2("6-well") == pytest.approx(9.6)


def test_medium_volume_per_well():
    assert culture.medium_volume_per_well("96") == pytest.approx(0.17)
    assert culture.medium_volume_per_well("6") == pytest.approx(2.7)


def test_medium_volume_per_area_override():
    # 0.32 cm^2 * 0.2 mL/cm^2 = 0.064 mL (low-volume 96-well protocol)
    assert culture.medium_volume_per_well("96", volume_per_area_ml_cm2=0.2) == pytest.approx(0.064)


def test_cells_per_well():
    # 2e4 cells/cm2 * 0.32 cm2 (96-well) = 6400
    assert culture.cells_per_well(2e4, "96") == pytest.approx(6400.0)
    # 1.5e5 * 9.6 (6-well) = 1.44e6
    assert culture.cells_per_well(1.5e5, "6") == pytest.approx(1.44e6)


def test_hemocytometer_count():
    # 32 cells/square, 1:2 dilution -> 32 * 2 * 1e4 = 6.4e5
    assert culture.hemocytometer_count(32, 2) == pytest.approx(6.4e5)
    assert culture.hemocytometer_count(20, 1) == pytest.approx(2e5)


def test_hemocytometer_dilution_ge_one():
    with pytest.raises(ValueError):
        culture.hemocytometer_count(32, 0.5)


def test_trypan_blue_viability():
    assert culture.trypan_blue_viability(15, 5) == pytest.approx(75.0)
    assert culture.trypan_blue_viability(0, 10) == pytest.approx(0.0)


def test_trypan_blue_viability_zero_total():
    with pytest.raises(ValueError):
        culture.trypan_blue_viability(0, 0)


def test_viable_cells_in_suspension():
    # 6.4e5 cells/mL, 1 mL, 90% -> 5.76e5
    assert culture.viable_cells_in_suspension(6.4e5, 1, 90) == pytest.approx(5.76e5)


def test_confluence_round_trip():
    # 50% confluence at 1e6/cm2 over 0.32 cm2 = 1.6e5 cells
    n = culture.confluence_to_cell_count(50, 1e6, 0.32)
    assert n == pytest.approx(1.6e5)
    assert culture.cell_count_to_confluence(n, 1e6, 0.32) == pytest.approx(50.0)


def test_confluence_out_of_range():
    with pytest.raises(ValueError):
        culture.confluence_to_cell_count(110, 1e6, 0.32)


def test_time_to_confluence_pct():
    # seed 3.8e4 in 24-well (1.9 cm2), 1e6/cm2 confluent, 72h at 30h doubling.
    # final = 3.8e4 * 2^(72/30) = 3.8e4 * 5.278 = 2.0056e5; confluence = 2.0056e5/(1e6*1.9)*100
    final = cell.cell_count_after_time(3.8e4, 30, 72)
    pct = culture.cell_count_to_confluence(final, 1e6, 1.9)
    assert culture.time_to_confluence_pct(3.8e4, 1e6, 1.9, pct, 30) == pytest.approx(72.0)


def test_time_to_confluence_pct_already_met():
    assert culture.time_to_confluence_pct(2e5, 1e6, 1.9, 10, 30) == 0.0


def test_passage_split_ratio():
    # harvest 1.92e5 from a 6-well (9.6 cm2), reseed 2e4/cm2 -> exactly 1.0
    assert culture.passage_split_ratio(1.92e5, 2e4, "6") == pytest.approx(1.0)
    # harvest 3.84e5 -> 2.0
    assert culture.passage_split_ratio(3.84e5, 2e4, "6") == pytest.approx(2.0)


def test_passage_split_impossible():
    with pytest.raises(ValueError):
        culture.passage_split_ratio(1e5, 2e4, "6")


def test_moi_virus_volume():
    # MOI 5 on 2e5 cells, titer 1e7 -> 5*2e5/1e7 = 0.1 mL
    assert culture.moi_virus_volume(5, 2e5, 1e7) == pytest.approx(0.1)


def test_cryo_vial_count():
    assert culture.cryo_vial_count(2.5e6, 1e6) == 3
    assert culture.cryo_vial_count(2e6, 1e6) == 2
