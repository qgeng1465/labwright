"""CPU-only tests for the SciRecipe audit (no GPU, no model download)."""

import pytest

from eval.run_scirecipe_audit import (
    audit_row,
    harvest_claims,
    has_numbers,
    is_culture,
    is_microfluidics,
    route_domain,
)


# ---------------------------------------------------------------------------
# Funnel
# ---------------------------------------------------------------------------


def test_has_numbers():
    assert has_numbers("Seeded 1e4 cells/cm2 in a 96-well plate at 37 °C")
    assert has_numbers("Perfused at 100 µL/min for 24 h")
    assert not has_numbers("Grow bacteria under both conditions")


def test_domain_routing():
    assert route_domain("Seed HepG2 at 1e4 cells/cm2 in a 96-well plate, incubate 37°C") == "culture"
    assert route_domain("Perfuse the 400 µm channel at 2 µL/min to reach 0.05 Pa shear") == "flow"
    assert route_domain("Grind samples and vortex for 5 min") == "none"
    # A culture row that mentions flow rate is still culture (plate signals win).
    assert route_domain("Seed in 24-well plates, perfuse media at 100 µL/min") == "culture"


def test_funnel_signal_helpers():
    assert is_culture("cells/cm2 seeding in 6-well plate")
    assert is_microfluidics("channel at a flow rate of 5 µL/min")


# ---------------------------------------------------------------------------
# Claim harvesting
# ---------------------------------------------------------------------------


def test_harvest_shear_only_with_context():
    assert harvest_claims("wall shear stress 0.05 Pa") == {"shear_pa": 0.05}
    assert harvest_claims("shear 5 dyn/cm2") == {"shear_pa": 0.5}
    # bare Pa without shear/pressure context is ambiguous -> not claimed
    assert "shear_pa" not in harvest_claims("pressure drop of 10 Pa")


def test_harvest_reynolds_and_culture():
    assert harvest_claims("Reynolds number 0.3") == {"reynolds": 0.3}
    claims = harvest_claims("seeded 3200 cells per well")
    assert claims["seed_per_well"] == 3200
    claims = harvest_claims("1.5 mL per well medium")
    assert claims["medium_volume_per_well_ml"] == 1.5
    claims = harvest_claims("80% confluence at day 3")
    assert claims["expected_confluence_pct"] == 80


# ---------------------------------------------------------------------------
# Row audit with a stub extractor
# ---------------------------------------------------------------------------


def _stub_extract(raw):
    def fn(_orc):
        return raw
    return fn


def test_audit_culture_consistent_row():
    orc = "Seed 1e4 cells/cm2 in a 96-well plate; seeded 3200 cells per well."
    raw = {"culture": {"plate_format": "96-well", "seeding_density_cells_cm2": 1e4}}
    rec = audit_row(orc, _stub_extract(raw), reference="ref-1")
    assert rec["verdict"] == "ok"
    assert rec["domain"] == "culture"
    assert rec["computed"]["seed_per_well"] == pytest.approx(3200, rel=1e-6)


def test_audit_culture_contradiction_row():
    orc = "Seed 1e4 cells/cm2 in a 96-well plate; seeded 6400 cells per well."
    raw = {"culture": {"plate_format": "96-well", "seeding_density_cells_cm2": 1e4}}
    rec = audit_row(orc, _stub_extract(raw), reference="ref-2")
    assert rec["verdict"] == "review_required"
    assert rec["discrepancy_fields"] == ["seed_per_well"]


def test_audit_flow_row():
    orc = "400 µm channel, shear 0.05 Pa."
    raw = {"chip": {"width_um": 400, "height_um": 100, "length_mm": 20},
           "flow": {"flow_rate_uLmin": 2, "viscosity_pas": 0.001}}
    rec = audit_row(orc, _stub_extract(raw), reference="ref-3")
    assert rec["verdict"] == "ok"
    assert rec["domain"] == "flow"
    assert rec["computed"]["shear_pa"] == pytest.approx(0.05, rel=1e-3)


def test_audit_no_domain_unverifiable():
    orc = "Grind samples and vortex for 5 min."
    rec = audit_row(orc, _stub_extract({}), reference="ref-4")
    assert rec["verdict"] == "unverifiable"
    assert rec["reason"] == "no_domain"


def test_audit_extract_failure_unverifiable():
    orc = "Seed 1e4 cells/cm2 in a 96-well plate."
    rec = audit_row(orc, lambda _: None, reference="ref-5")
    assert rec["verdict"] == "unverifiable"
    assert rec["reason"] == "extract_failed"


def test_audit_domain_raw_mismatch_unverifiable():
    # routed culture, but the extractor returned only flow raws
    orc = "Seed 1e4 cells/cm2 in a 96-well plate."
    raw = {"chip": {"width_um": 400}, "flow": {"flow_rate_uLmin": 2}}
    rec = audit_row(orc, _stub_extract(raw), reference="ref-6")
    assert rec["verdict"] == "unverifiable"
    assert rec["reason"] == "no_culture_raw"
