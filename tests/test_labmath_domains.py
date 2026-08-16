"""Tests for the five LabMath-Bench calc modules and their full-chain wiring.

The reviewer-facing LabMath-Bench asks for three difficulty levels:
L1 fluid/space engineering (bioprinting), L2 biochemical stoichiometry
(co-culture, enzyme competitive binding) and L3 pipeline parameterization
(ChAMP / PLINK batches, solvent evaporation). Each new calc module is tested
against its formula here, and the full-chain wiring is pinned: a ``Block``
declaration, a ``DesignInput``/``derive_*`` path that produces verified plans,
a checker that re-derives every number, sanity bands that flag out-of-band
values, provenance records with formula+inputs+units+status, registered tools,
and benchmark routing for both the bare and design scoring paths.
"""

import pytest

from labwright.calc import bioinformatics, bioprinting, coculture, enzyme, solvent
from labwright.design import DesignInput, submit_design
from labwright.schema.design import DesignPlan
from labwright.sop.provenance import provenance_for
from labwright.verify.sanity import check_sanity
from labwright.verify.checker import Issue, verify_design


# ---------------------------------------------------------------------------
# bioprinting (L1) — micro-extrusion geometry / kinematics
# ---------------------------------------------------------------------------


def test_nozzle_table_resolution():
    assert bioprinting.nozzle_diameter_um("nozzle_3") == pytest.approx(500.0)
    assert bioprinting.nozzle_diameter_um("3") == pytest.approx(500.0)
    assert bioprinting.nozzle_diameter_um("cryo3") == pytest.approx(500.0)
    assert bioprinting.nozzle_diameter_um("uv5") == pytest.approx(300.0)
    assert bioprinting.nozzle_kind("4") == "photocuring"
    with pytest.raises(ValueError):
        bioprinting.nozzle_diameter_um("nozzle_99")


def test_extrusion_volume_from_nozzle_and_travel():
    # V = π(d/2)²·L; 500 µm nozzle over 10 mm → 1963.5 nL
    assert bioprinting.extrusion_volume_nl(10000.0, 500.0) == pytest.approx(1963.4954, rel=1e-4)
    # same path through a 300 µm UV nozzle deposits less
    assert bioprinting.extrusion_volume_nl(10000.0, 300.0) == pytest.approx(706.8583, rel=1e-4)
    with pytest.raises(ValueError):
        bioprinting.extrusion_volume_nl(0, 500.0)


def test_print_time_and_rate():
    # t = L/v: 10 mm at 5 mm/min = 2 min = 120 s
    assert bioprinting.print_time_s(10000.0, 5.0) == pytest.approx(120.0)
    vol = bioprinting.extrusion_volume_nl(10000.0, 500.0)
    assert bioprinting.extrusion_rate_nl_min(vol, 120.0) == pytest.approx(981.7477, rel=1e-4)


def test_filament_mass_and_lines():
    # m = ρ·V: 1963 nL at 1.05 g/cm³ ≈ 2061.7 µg
    vol = bioprinting.extrusion_volume_nl(10000.0, 500.0)
    assert bioprinting.filament_mass_ug(vol, 1.05) == pytest.approx(2061.6702, rel=1e-4)
    # 2000 µm footprint at 400 µm pitch → 5 lines
    assert bioprinting.lines_to_cover(2000.0, 400.0) == 5
    assert bioprinting.lines_to_cover(2000.0, 600.0) == 4  # ceil


def test_path_length_from_offset():
    # G-code offset (8000, 6000) µm → 10 mm diagonal
    assert bioprinting.path_length_from_offset(8000.0, 6000.0) == pytest.approx(10000.0)


# ---------------------------------------------------------------------------
# coculture (L2) — two-population seeding stoichiometry
# ---------------------------------------------------------------------------


def test_coculture_cells_from_fraction():
    a, b = coculture.cells_from_fraction(10000.0, 0.25)
    assert a == pytest.approx(2500.0)
    assert b == pytest.approx(7500.0)


def test_coculture_cells_per_well():
    # 5e4 cells/cm² × 0.32 cm² = 1.6e4/well; 25% A → 4000 A, 12000 B
    a, b = coculture.cells_per_well(5e4, 0.32, 0.25)
    assert a == pytest.approx(4000.0)
    assert b == pytest.approx(12000.0)


def test_coculture_total_and_ratio():
    assert coculture.total_cells(4000.0, 6) == pytest.approx(24000.0)
    assert coculture.total_for_two_wells(4000.0, 12000.0, 6) == pytest.approx(96000.0)
    assert coculture.seeding_ratio(4000.0, 12000.0) == pytest.approx(0.3333, rel=1e-3)
    with pytest.raises(ValueError):
        coculture.seeding_ratio(4000.0, 0.0)


# ---------------------------------------------------------------------------
# enzyme (L2) — competitive inhibition (Cheng–Prusoff)
# ---------------------------------------------------------------------------


def test_enzyme_fractional_activity():
    # v_i/v_0 = [S]/(Km(1+[I]/Ki)+[S]) = 180/(180(1+60/25)+180) = 0.2273
    assert enzyme.fractional_activity(180.0, 180.0, 25.0, 60.0) == pytest.approx(0.2273, rel=1e-3)
    # no inhibitor → activity 1.0
    # no inhibitor → the plain Michaelis-Menten fraction [S]/(Km+[S]); at
    # [S] = Km that is exactly 0.5
    assert enzyme.fractional_activity(180.0, 180.0, 25.0, 0.0) == pytest.approx(0.5)
    assert enzyme.fractional_activity(180.0, 1800.0, 25.0, 0.0) == pytest.approx(1800.0 / 1980.0)


def test_enzyme_percent_inhibition():
    assert enzyme.percent_inhibition(0.2273) == pytest.approx(77.27, rel=1e-3)
    with pytest.raises(ValueError):
        enzyme.percent_inhibition(1.5)


def test_enzyme_cheng_prusoff_ic50():
    # IC50 = Ki(1 + [S]/Km) = 25(1 + 180/180) = 50
    assert enzyme.ic50_from_ki(180.0, 180.0, 25.0) == pytest.approx(50.0)
    # Ki recovered from the run-condition IC50 returns the intrinsic Ki
    assert enzyme.ki_from_ic50(180.0, 180.0, 50.0) == pytest.approx(25.0)


def test_enzyme_apparent_km():
    # Km^app = Km(1 + [I]/Ki) = 180(1 + 60/25) = 612
    assert enzyme.apparent_km_um(180.0, 60.0, 25.0) == pytest.approx(612.0)


def test_enzyme_velocity_and_ratio():
    assert enzyme.velocity_umol_min(0.5, 180.0, 180.0, 25.0, 60.0) == pytest.approx(0.11364, rel=1e-3)
    assert enzyme.molar_ratio(60.0, 180.0) == pytest.approx(0.3333, rel=1e-3)


# ---------------------------------------------------------------------------
# bioinformatics (L3) — ChAMP / PLINK batch parameterization
# ---------------------------------------------------------------------------


def test_champ_batch_sizing():
    assert bioinformatics.champ_arrays_for_samples(98, "450k") == 98
    assert bioinformatics.champ_chips_for_samples(98, "450k") == 9  # ceil(98/12)
    assert bioinformatics.champ_chips_for_samples(98, "epic") == 13  # ceil(98/8)
    assert bioinformatics.champ_expected_failed_arrays(98, 0.05) == pytest.approx(4.9)
    with pytest.raises(ValueError):
        bioinformatics.champ_chips_for_samples(98, "unknown")


def test_plink_bed_sizing():
    # 800 × 2.4e6 / 4 / 1e6 = 480 MB
    assert bioinformatics.plink_bed_size_mb(800, 2_400_000) == pytest.approx(480.0)
    assert bioinformatics.plink_per_chr_files() == 25
    assert bioinformatics.plink_per_chr_bed_size_mb(800, 120_000) == pytest.approx(24.0)


# ---------------------------------------------------------------------------
# solvent (L3) — Langmuir d²-law evaporation + edge effect
# ---------------------------------------------------------------------------


def test_solvent_saturation_and_rate():
    # Magnus at 25 °C → ~23 g/m³ (documented set-point)
    assert solvent.saturation_conc_g_m3(25.0) == pytest.approx(22.98, rel=1e-2)
    # a 2 µL drop at 25 °C / 60% RH interior-evaporates at ~7.8 µL/hr
    assert solvent.evaporation_rate_ul_hr(2.0, 25.0, 0.60) == pytest.approx(7.82, rel=1e-2)


def test_solvent_drop_volume_dlaw():
    # d²-law slows as the drop shrinks: after 0.2 h interior, ~0.66 µL remains
    v = solvent.drop_volume_after_time(2.0, 0.2, 25.0, 0.60, evaporation_factor=1.0)
    assert v == pytest.approx(0.6620, rel=1e-2)
    # edge well (factor 1.5) dries faster
    v_edge = solvent.drop_volume_after_time(2.0, 0.2, 25.0, 0.60, evaporation_factor=1.5)
    assert v_edge < v


def test_solvent_edge_well_factor():
    assert solvent.edge_well_factor("A", 1) == pytest.approx(1.5)
    assert solvent.edge_well_factor("D", 6) == pytest.approx(1.0)
    assert solvent.edge_well_factor("H", 12) == pytest.approx(1.5)
    with pytest.raises(ValueError):
        solvent.edge_well_factor("I", 1)


def test_solvent_effective_rate():
    rate = solvent.effective_evaporation_rate_ul_hr(2.0, "A", 1, 25.0, 0.60)
    assert rate == pytest.approx(11.73, rel=1e-2)  # 7.82 × 1.5


# ---------------------------------------------------------------------------
# Full-chain wiring: block → schema → derive → check → sanity → provenance
# ---------------------------------------------------------------------------


def test_submit_design_derives_all_new_domains():
    r = submit_design({
        "goal": "Extrude a filament across a 10 mm path with the cryo-3 nozzle; co-seed "
                "HUVEC-T1/HepG2 at 25% A; quantify OA inhibition of UDPGA conjugation; "
                "size a 450k batch for 98 samples; size PLINK for 800×2.4e6; track a "
                "2 µL A1 hanging drop for 12 min.",
        "rationale": "combined LabMath-Bench smoke",
        "bioprinting": {"nozzle_id": "3", "travel_distance_um": 10000, "feed_rate_mm_min": 5.0,
                        "density_g_cm3": 1.05, "footprint_width_um": 2000, "line_pitch_um": 400},
        "coculture": {"cell_type_a": "HUVEC-T1", "cell_type_b": "HepG2",
                      "total_density_cells_cm2": 5e4, "area_cm2": 0.32, "fraction_a": 0.25, "wells": 6},
        "enzyme": {"enzyme": "UGT2B7", "substrate": "UDPGA", "km_um": 180.0, "s_conc_um": 180.0,
                   "ki_um": 25.0, "i_conc_um": 60.0, "vmax_umol_min": 0.5},
        "champ": {"n_samples": 98, "platform": "450k", "fail_rate_pct": 5.0},
        "plink": {"n_samples": 800, "n_variants": 2_400_000, "n_variants_chr": 120_000},
        "solvent": {"drop_volume_ul": 2.0, "hours": 0.2, "temp_c": 25.0, "rh": 0.60,
                    "well_row": "A", "well_col": 1},
    })
    assert r["status"] == "ok", r["verification_summary"]
    d = r["design"]
    assert d["bioprinting"]["extrusion_volume_nl"] == pytest.approx(1963.4954, rel=1e-4)
    assert d["coculture"]["seeding_ratio_ab"] == pytest.approx(0.3333, rel=1e-3)
    assert d["enzyme"]["ic50_um"] == pytest.approx(50.0, rel=1e-3)
    assert d["champ"]["n_chips"] == 9
    assert d["plink"]["bed_size_mb"] == pytest.approx(480.0)
    assert d["solvent"]["edge_evaporation_factor"] == pytest.approx(1.5)


def test_derived_fields_rejected_from_model_input():
    with pytest.raises(ValueError, match="derived field"):
        submit_design({
            "goal": "g", "rationale": "r",
            "bioprinting": {"nozzle_id": "3", "travel_distance_um": 10000,
                            "feed_rate_mm_min": 5.0, "extrusion_volume_nl": 999.0},
        })
    with pytest.raises(ValueError, match="derived field"):
        submit_design({
            "goal": "g", "rationale": "r",
            "enzyme": {"km_um": 180.0, "s_conc_um": 180.0, "ki_um": 25.0,
                       "i_conc_um": 60.0, "fractional_activity": 0.5},
        })


def test_verifier_catches_tampered_derived_value():
    # Build the raw input, then re-derive by hand and corrupt one derived field.
    raw = {
        "goal": "g", "rationale": "r",
        "enzyme": {"enzyme": "E", "substrate": "S", "km_um": 180.0, "s_conc_um": 180.0,
                   "ki_um": 25.0, "i_conc_um": 60.0},
    }
    plan = submit_design(raw, verify=False)["design"]
    plan["enzyme"]["ic50_um"] = 123.0  # model smuggled a wrong number
    dp = DesignPlan(**plan)
    issues = verify_design(dp)
    assert any(i.field == "enzyme.ic50_um" and i.level == "error" for i in issues)


def test_sanity_bands_flag_out_of_range():
    # enzyme i_conc_um band soft max 1e5; a 1e8 µM inhibitor is a hard error
    raw = {
        "goal": "g", "rationale": "r",
        "enzyme": {"enzyme": "E", "substrate": "S", "km_um": 180.0, "s_conc_um": 180.0,
                   "ki_um": 25.0, "i_conc_um": 1e8},
    }
    plan = submit_design(raw, verify=False)["design"]
    dp = DesignPlan(**plan)
    issues: list[Issue] = []
    check_sanity(dp, issues)
    assert any(i.field == "enzyme.i_conc_um" and i.level == "error" for i in issues)


def test_provenance_records_for_new_domains():
    r = submit_design({
        "goal": "g", "rationale": "r",
        "enzyme": {"enzyme": "UGT2B7", "substrate": "UDPGA", "km_um": 180.0, "s_conc_um": 180.0,
                   "ki_um": 25.0, "i_conc_um": 60.0, "vmax_umol_min": 0.5},
        "solvent": {"drop_volume_ul": 2.0, "hours": 0.2, "temp_c": 25.0, "rh": 0.60,
                    "well_row": "A", "well_col": 1},
    })
    plan = DesignPlan(**r["design"])
    issues = verify_design(plan)
    records = provenance_for(plan, issues)
    fields = {rec["field"] for rec in records}
    assert "enzyme.fractional_activity" in fields
    assert "enzyme.ic50_um" in fields
    assert "solvent.residual_volume_ul" in fields
    for rec in records:
        assert rec["formula"], rec["field"]
        assert rec["inputs"], rec["field"]
        assert rec["unit"], rec["field"]
        assert rec["status"] in ("ok", "warning", "error")


def test_new_tools_registered():
    import labwright.tools as T

    for name in (
        "bioprinting_extrusion_volume_nl", "bioprinting_path_length",
        "coculture_cells_per_well", "coculture_seeding_ratio",
        "enzyme_fractional_activity", "enzyme_ic50_from_ki",
        "champ_chips_for_samples", "plink_bed_size_mb",
        "solvent_evaporation_rate", "solvent_drop_volume_after_time",
    ):
        assert name in T.REGISTRY, name


def test_benchmark_routing_for_new_domains():
    from types import SimpleNamespace

    import eval.benchmark as B

    # a new-domain gold resolves to the right block by derived-key overlap
    for expected, block in (
        ({"extrusion_volume_nl": 1.0}, "bioprinting"),
        ({"cells_per_well_a": 1.0, "seeding_ratio_ab": 0.5}, "coculture"),
        ({"fractional_activity": 0.5, "ic50_um": 50.0}, "enzyme"),
        ({"n_arrays": 98, "n_chips": 9}, "champ"),
        ({"bed_size_mb": 480.0}, "plink"),
        ({"evaporation_rate_ul_hr": 11.7}, "solvent"),
    ):
        assert B._new_domain_block(SimpleNamespace(expected=expected)) == block

    # _new_domain_computed recomputes derived numbers from the reported raws
    computed = B._new_domain_computed(
        {"km_um": 180.0, "s_conc_um": 180.0, "ki_um": 25.0, "i_conc_um": 60.0}, "enzyme"
    )
    assert computed["ic50_um"] == pytest.approx(50.0, rel=1e-3)

    # bare_checkable routes an enzyme raw report to the enzyme block
    assert B.bare_checkable(
        {"km_um": 180.0, "s_conc_um": 180.0, "ki_um": 25.0, "i_conc_um": 60.0,
         "fractional_activity": 0.2273}
    )
    assert not B.bare_checkable({"km_um": 180.0})

    # the design-side claims loop reports enzyme values with bare gold keys
    r = submit_design({
        "goal": "g", "rationale": "r",
        "enzyme": {"enzyme": "E", "substrate": "S", "km_um": 180.0, "s_conc_um": 180.0,
                   "ki_um": 25.0, "i_conc_um": 60.0},
    })
    plan = DesignPlan(**r["design"])
    claims = B._new_domain_claims(plan)
    assert claims["ic50_um"] == pytest.approx(50.0, rel=1e-3)
