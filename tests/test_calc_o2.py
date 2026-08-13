"""Tests for the O2 transport calculators and their verifier hook.

The physics (Henry's law, Krogh penetration, necrotic-core fraction, Péclet,
Damköhler, supply-vs-demand) must agree with the textbook values to machine
precision, and ``check_oxygen`` must warn on a perfused design whose supply
cannot meet demand — but stay silent when it has no right to a number (no flow,
or no registry OCR for the cell type).
"""

import math

import pytest

from labwright.calc import o2 as c
from labwright.design import build_design, DesignInput
from labwright.tools import REGISTRY
from labwright.verify.checker import verify_design


# -- Henry's law ------------------------------------------------------------


def test_henry_forward_air_saturation():
    # 150 mmHg → ~0.2 mM: the canonical "air-equilibrated medium ≈ 200 µM".
    assert c.o2_conc_mm_from_po2(150) == pytest.approx(0.201, abs=1e-3)


def test_henry_reverse_air_saturation():
    # 0.2 mM / (1.34 µmol/L/mmHg × 1e-3) = 149.25 mmHg.
    assert c.o2_po2_mmhg_from_conc(0.2) == pytest.approx(0.2 / (c.O2_HENRY_UMOL_L_MMHG * 1e-3), rel=1e-9)


def test_henry_roundtrip():
    assert c.o2_po2_mmhg_from_conc(c.o2_conc_mm_from_po2(120)) == pytest.approx(120, rel=1e-9)


# -- Krogh penetration -------------------------------------------------------


def test_volumetric_consumption_unit_math():
    # 0.03 fmol/s/cell × 1e8 cells/mL → mol/m³/s (1 mL = 1e-6 m³).
    q = c.volumetric_o2_consumption(0.03, 1e8)
    assert q == pytest.approx(0.03e-15 * 1e8 * 1e6, rel=1e-9)


def test_penetration_depth_dense_tissue():
    # q = 0.1 mol/m³/s, C0 = 0.2 mol/m³, D = 2e-9 → ~89 µm (tens of µm).
    d = c.o2_penetration_depth_um(0.1)
    assert d == pytest.approx(math.sqrt(2 * 2e-9 * 0.2 / 0.1) * 1e6, rel=1e-9)
    assert 80 < d < 100


def test_penetration_depth_gives_the_200um_rule():
    # The empirical "oxygen diffuses ~200 µm" rule corresponds to q ≈ 0.02.
    d = c.o2_penetration_depth_um(0.02)
    assert d == pytest.approx(200.0, rel=0.05)


# -- Necrotic core ------------------------------------------------------------


def test_necrotic_fraction_known_value():
    # 500 µm spheroid (R = 250), 200 µm penetration → (1 - 0.8)³ = 0.008.
    assert c.spheroid_necrotic_fraction(500, 200) == pytest.approx(0.008, rel=1e-9)


def test_necrotic_fraction_zero_when_fully_oxygenated():
    # 200 µm spheroid with 200 µm penetration: δ ≥ R → 0.
    assert c.spheroid_necrotic_fraction(200, 200) == 0.0
    assert c.spheroid_necrotic_fraction(200, 150) == 0.0  # δ = R


def test_necrotic_fraction_monotonic_in_diameter():
    small = c.spheroid_necrotic_fraction(300, 100)
    large = c.spheroid_necrotic_fraction(800, 100)
    assert 0 <= small < large < 1


# -- Demand and dimensionless numbers -----------------------------------------


def test_demand_unit_math():
    # 1e6 cells at 0.03 fmol/s → µmol/min.
    assert c.o2_demand_umol_min(1e6, 0.03) == pytest.approx(1.8e-3, rel=1e-9)


def test_ocr_conversion_nmol_min_to_fmol_s():
    # 1 nmol/min per 10⁶ cells = 1 fmol/min per cell = 1/60 fmol/s per cell.
    assert c.nmol_min_per_1e6_to_fmol_s(1.0) == pytest.approx(1.0 / 60, rel=1e-9)
    assert c.nmol_min_per_1e6_to_fmol_s(3.0) == pytest.approx(0.05, rel=1e-9)


def test_peclet_number():
    # u = 0.5 mm/s over L = 20 mm, D = 2e-9 → Pe = (5e-4)(2e-2)/2e-9 = 5000.
    assert c.peclet_number(0.5, 20) == pytest.approx(5000.0, rel=1e-9)


def test_damkohler_number():
    # q·L/(u·C0) = 0.1·0.02/(5e-4·0.2) = 20.
    assert c.damkohler_number(0.1, 20, 0.5) == pytest.approx(20.0, rel=1e-9)


def test_peclet_dimensionless_scale():
    # Slow flow, short channel → diffusion dominates (Pe < 1).
    assert c.peclet_number(0.001, 0.2) < 1


# -- Supply vs demand ---------------------------------------------------------


def test_supply_vs_demand_ample_flow():
    out = c.o2_supply_vs_demand(50, 1e6, 0.03)  # supply 0.01, demand 0.0018
    assert out["supply_umol_min"] == pytest.approx(0.01, rel=1e-6)
    assert out["demand_umol_min"] == pytest.approx(1.8e-3, rel=1e-6)
    assert out["ratio"] == pytest.approx(0.01 / 1.8e-3, rel=1e-3)
    assert out["hypoxic"] is False


def test_supply_vs_demand_hypoxic():
    out = c.o2_supply_vs_demand(5, 1e7, 0.05)  # supply 0.001, demand 0.03
    assert out["hypoxic"] is True


# -- Validation ---------------------------------------------------------------


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        c.o2_conc_mm_from_po2(-10)
    with pytest.raises(ValueError):
        c.o2_po2_mmhg_from_conc(-0.1)
    with pytest.raises(ValueError):
        c.o2_penetration_depth_um(0)
    with pytest.raises(ValueError):
        c.spheroid_necrotic_fraction(0, 100)
    with pytest.raises(ValueError):
        c.peclet_number(-1, 20)
    with pytest.raises(ValueError):
        c.damkohler_number(0, 20, 0.5)


# -- Tool registration --------------------------------------------------------


def test_o2_tools_registered():
    for name in ("o2_penetration_depth", "spheroid_necrotic_fraction",
                 "o2_supply_vs_demand", "o2_peclet", "o2_damkohler",
                 "o2_po2_conversion"):
        assert name in REGISTRY, name


def test_o2_po2_conversion_tool_bidirectional():
    fwd = REGISTRY["o2_po2_conversion"].func(po2_mmHg=150)
    assert fwd["conc_mM"] == pytest.approx(0.201, abs=1e-3)
    rev = REGISTRY["o2_po2_conversion"].func(conc_mM=0.2)
    assert rev["po2_mmHg"] == pytest.approx(149.3, abs=0.1)  # tool rounds to 1 dp
    both = REGISTRY["o2_po2_conversion"].func(po2_mmHg=150, conc_mM=0.2)
    assert "error" in both


def test_o2_penetration_tool_matches_module():
    out = REGISTRY["o2_penetration_depth"].func(volumetric_consumption_mol_m3s=0.1)
    assert out == pytest.approx(c.o2_penetration_depth_um(0.1), rel=1e-9)


# -- check_oxygen verifier hook -----------------------------------------------


def _perfused_plan(cell_type="HepG2", seed_count=1e5, flow=10.0):
    """A perfused 2D chip design. HepG2 registry OCR = 0.8-4.7 nmol/min/10⁶.

    seed_count is derived (density × area) — derive the density from the target.
    """
    area = 0.08
    density = seed_count / area
    return build_design(DesignInput(
        goal="test", rationale="r",
        chip=dict(width_um=400, height_um=100, length_mm=20),
        flow=dict(flow_rate_uLmin=flow, viscosity_pas=1e-3),
        cells=dict(cell_type=cell_type, seeding_density_cells_cm2=density,
                   culture_area_cm2=area),
    ))


def _oxygen_issues(plan):
    return [i for i in verify_design(plan) if i.field == "flow.flow_rate_uLmin"
            and "perfused O2 supply" in i.message]


def test_check_oxygen_warns_on_hypoxic_perfused_design():
    # HepG2 mid OCR 2.75 nmol/min/10⁶ = 0.0458 fmol/s/cell.
    # At flow 1 µL/min supply = 1e-6 L/min × 0.2e-3 mol/L = 2e-4 µmol/min;
    # demand for 1e7 cells = 1e7 × 0.0458 × 6e-14 = 2.75e-8 mol/min = 0.0275 µmol/min.
    plan = _perfused_plan(seed_count=1e7, flow=1.0)
    issues = _oxygen_issues(plan)
    assert len(issues) == 1
    assert issues[0].level == "warning"


def test_check_oxygen_silent_when_well_supplied():
    # 1e5 HepG2 (demand ≈ 2.75e-4 µmol/min) at 10 µL/min (supply 2e-3) → fine.
    plan = _perfused_plan(seed_count=1e5, flow=10.0)
    assert _oxygen_issues(plan) == []


def test_check_oxygen_silent_without_flow():
    plan = build_design(DesignInput(
        goal="test", rationale="r",
        cells=dict(cell_type="HepG2", seeding_density_cells_cm2=1e5, culture_area_cm2=0.08),
    ))
    assert _oxygen_issues(plan) == []


def test_check_oxygen_silent_for_unknown_cell_type():
    # No registry OCR → the verifier must not invent demand.
    plan = _perfused_plan(cell_type="my-custom-line", seed_count=1e7, flow=1.0)
    assert _oxygen_issues(plan) == []
