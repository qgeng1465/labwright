"""Unit tests for labwright.calc.cell."""

import pytest

from labwright.calc import cell


def test_seeding_cell_count():
    # 1e5 cells/cm2 on 0.016 cm2
    assert cell.seeding_cell_count(1e5, 0.016) == pytest.approx(1600)


def test_culture_area():
    # 400 um wide, 20 mm long -> 0.04 cm x 2 cm = 0.08 cm2
    assert cell.culture_area(400, 20) == pytest.approx(0.08)


def test_exponential_growth():
    # 1000 cells, 24h doubling, 24h elapsed -> 2000
    assert cell.cell_count_after_time(1000, 24, 24) == pytest.approx(2000)


def test_growth_no_elapsed_time():
    assert cell.cell_count_after_time(1000, 24, 0) == pytest.approx(1000)


def test_time_to_confluence():
    # 1e4 -> 4e4 with 24h doubling: log2(4)*24 = 48h
    assert cell.time_to_confluence(1e4, 4e4, 24) == pytest.approx(48.0)


def test_time_to_confluence_already_met():
    assert cell.time_to_confluence(4e4, 4e4, 24) == 0.0
    assert cell.time_to_confluence(5e4, 4e4, 24) == 0.0


def test_expansion_factor():
    assert cell.required_expansion_factor(3e6, 1e5) == pytest.approx(30)


def test_viable_cells():
    assert cell.viable_cells(1000, 90) == pytest.approx(900)


def test_viability_out_of_range():
    with pytest.raises(ValueError):
        cell.viable_cells(1000, 101)
