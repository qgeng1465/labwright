"""Unit tests for labwright.calc.spheroid."""

import pytest

from labwright.calc import spheroid


def test_spheroid_formats_complete():
    assert set(spheroid.SPHEROID_FORMATS) == {"96-ula", "384-ula", "hanging-drop"}


def test_normalize_format():
    assert spheroid._normalize_format("96-ula") == "96-ula"
    assert spheroid._normalize_format("96well ULA") == "96-ula"
    assert spheroid._normalize_format("384-ula") == "384-ula"
    assert spheroid._normalize_format("hanging drop") == "hanging-drop"
    # "96 well ULA" / "96-well ULA plate" — the phrasings a bare LLM actually
    # writes — must normalise too, or a correct format string is scored
    # unverifiable in the benchmark.
    assert spheroid._normalize_format("96 well ULA") == "96-ula"
    assert spheroid._normalize_format("96-well ULA plate") == "96-ula"
    assert spheroid._normalize_format("384-well ULA") == "384-ula"
    with pytest.raises(ValueError):
        spheroid._normalize_format("petri-dish")


def test_spheroid_volume_ul():
    # V = 4/3*pi*(100 um)^3 = 4.18879e6 um^3 = 4.18879e-3 uL
    assert spheroid.spheroid_volume_ul(200.0) == pytest.approx(4.18879e-3)
    # a 300 um spheroid ≈ 0.01414 uL ≈ 14 nL
    assert spheroid.spheroid_volume_ul(300.0) == pytest.approx(1.4137e-2, rel=1e-3)


def test_spheroid_diameter_um_round_trip():
    v = spheroid.spheroid_volume_ul(200.0)
    assert spheroid.spheroid_diameter_um(v) == pytest.approx(200.0)


def test_spheroid_surface_area_mm2():
    # A = 4*pi*(100 um)^2 = 125663.7 um^2 = 0.12566 mm^2
    assert spheroid.spheroid_surface_area_mm2(200.0) == pytest.approx(0.125664, rel=1e-4)


def test_cell_volume_ul():
    # a 20 um hepatocyte ≈ 4.2 pL
    assert spheroid.cell_volume_ul(20.0) == pytest.approx(4.18879e-6)


def test_spheroid_volume_from_cells():
    # 1000 cells of 20 um packed solid = 1000 * 4.18879e-6 = 4.18879e-3 uL
    assert spheroid.spheroid_volume_from_cells(1000.0, 20.0) == pytest.approx(4.18879e-3)


def test_spheroid_diameter_from_cells():
    # 1000 cells of 20 um -> ~200 um spheroid
    assert spheroid.spheroid_diameter_from_cells(1000.0, 20.0) == pytest.approx(200.0)
    # 100 cells -> ~93 um
    assert spheroid.spheroid_diameter_from_cells(100.0, 20.0) == pytest.approx(92.82, rel=1e-3)


def test_cells_per_spheroid_for_diameter():
    # V(200 um) / V(20 um cell) = 1000
    assert spheroid.cells_per_spheroid_for_diameter(200.0, 20.0) == pytest.approx(1000.0)


def test_spheroid_count_from_suspension():
    assert spheroid.spheroid_count_from_suspension(2.4e5, 1000.0) == 240
    assert spheroid.spheroid_count_from_suspension(2.4e5 + 999, 1000.0) == 240  # floor
    assert spheroid.spheroid_count_from_suspension(999, 1000.0) == 0


def test_cells_needed_for_spheroids():
    assert spheroid.cells_needed_for_spheroids(96, 1000.0) == pytest.approx(96000.0)


def test_medium_volume_per_spheroid():
    assert spheroid.medium_volume_per_spheroid("96-ula") == pytest.approx(100.0)
    assert spheroid.medium_volume_per_spheroid("384-ula") == pytest.approx(50.0)
    assert spheroid.medium_volume_per_spheroid("hanging-drop") == pytest.approx(20.0)


def test_total_medium_volume():
    # 96 spheroids at 100 uL = 9.6 mL
    assert spheroid.total_medium_volume(96, 100.0) == pytest.approx(9.6)
    # 48 hanging drops at 20 uL = 0.96 mL
    assert spheroid.total_medium_volume(48, 20.0) == pytest.approx(0.96)


def test_total_medium_volume_adds_dead_volume():
    # Reservoir + tubing on a perfusion circuit holds ~2 mL that never reaches a well.
    assert spheroid.total_medium_volume(96, 100.0, dead_volume_ul=2000.0) == pytest.approx(11.6)
    assert spheroid.total_medium_volume(96, 100.0, dead_volume_ul=0.0) == pytest.approx(9.6)
    with pytest.raises(ValueError):
        spheroid.total_medium_volume(96, 100.0, dead_volume_ul=-1.0)


def test_packing_fraction_loosens_spheroid():
    """Real (looser-than-solid) packing makes a spheroid larger / need fewer cells."""
    # default 1.0 keeps the "1000 cells of 20 um -> 200 um" convention
    assert spheroid.spheroid_volume_from_cells(1000.0, 20.0) == pytest.approx(4.18879e-3)
    # packing 0.74 -> volume 1/0.74 times larger
    assert spheroid.spheroid_volume_from_cells(1000.0, 20.0, packing_fraction=0.74) == pytest.approx(
        4.18879e-3 / 0.74, rel=1e-4
    )
    # diameter grows: 1000 cells of 20 um at packing 0.74 -> ~222 um
    assert spheroid.spheroid_diameter_from_cells(1000.0, 20.0, packing_fraction=0.74) == pytest.approx(
        200.0 / (0.74 ** (1 / 3)), rel=1e-4
    )
    # fewer cells reach the same target diameter
    assert spheroid.cells_per_spheroid_for_diameter(200.0, 20.0, packing_fraction=0.74) == pytest.approx(740.0, rel=1e-3)
    with pytest.raises(ValueError):
        spheroid.spheroid_diameter_from_cells(1000.0, 20.0, packing_fraction=1.5)


def test_validate_positive():
    with pytest.raises(ValueError):
        spheroid.spheroid_volume_ul(0.0)
    with pytest.raises(ValueError):
        spheroid.spheroid_volume_ul(-5.0)
    with pytest.raises(ValueError):
        spheroid.spheroid_diameter_from_cells(0.0, 20.0)
