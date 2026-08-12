"""Regression tests for the culture block added to published-protocol checks."""

import pytest

from labwright.published import CULTURE_METRICS, verify_published_protocol


def test_culture_seed_per_well_consistent():
    # 1e4 cells/cm^2 in a 96-well plate (0.32 cm^2) -> 3200 cells/well, 0.17 mL/well.
    result = verify_published_protocol(
        culture={"plate_format": "96-well", "seeding_density_cells_cm2": 1e4},
        claimed={"seed_per_well": 3200, "medium_volume_per_well_ml": 0.17},
        reference="10.1000/culture-self-consistent",
    )
    assert result["status"] == "ok"
    verdicts = {c["field"]: c["verdict"] for c in result["checks"]}
    assert verdicts["seed_per_well"] == "consistent"
    assert verdicts["medium_volume_per_well_ml"] == "consistent"
    assert result["n_discrepancies"] == 0


def test_culture_fluid_mix_checks_both_domains():
    result = verify_published_protocol(
        chip={"width_um": 400, "height_um": 100, "length_mm": 20},
        flow={"flow_rate_uLmin": 2, "viscosity_pas": 0.001},
        culture={"plate_format": "6-well", "seeding_density_cells_cm2": 1.5e5},
        claimed={"shear_pa": 0.05, "seed_per_well": 1.44e6},
        reference="10.1000/mixed",
    )
    assert result["status"] == "ok"
    fields = {c["field"] for c in result["checks"]}
    assert "shear_pa" in fields and "seed_per_well" in fields


def test_culture_wrong_seed_per_well_flagged():
    result = verify_published_protocol(
        culture={"plate_format": "96-well", "seeding_density_cells_cm2": 1e4},
        claimed={"seed_per_well": 6400},  # 2x the true 3200
        reference="10.1000/culture-suspicious",
    )
    assert result["status"] == "review_required"
    assert result["n_discrepancies"] == 1
    seed = next(c for c in result["checks"] if c["field"] == "seed_per_well")
    assert seed["verdict"] == "discrepancy"


def test_culture_confluence_without_growth_inputs_is_unverifiable():
    # The paper claims confluence but does not report doubling time / confluent
    # density / culture duration -> we cannot recompute; the claim is neither
    # confirmed nor contradicted.
    result = verify_published_protocol(
        culture={"plate_format": "96-well", "seeding_density_cells_cm2": 1e4},
        claimed={"expected_confluence_pct": 80},
        reference="10.1000/culture-unverifiable",
    )
    assert result["status"] == "unverifiable"
    conf = next(c for c in result["checks"] if c["field"] == "expected_confluence_pct")
    assert conf["verdict"] == "unverifiable"


def test_culture_confluence_with_growth_inputs_consistent():
    result = verify_published_protocol(
        culture={
            "plate_format": "96-well",
            "seeding_density_cells_cm2": 1e4,
            "confluent_density_cells_cm2": 1.5e5,
            "doubling_time_h": 35,
            "culture_duration_h": 96,
        },
        claimed={"seed_per_well": 3200},
        reference="10.1000/culture-growth",
    )
    # seed_per_well checks out; confluence is computable and unclaimed.
    assert result["status"] == "ok"
    conf = next(c for c in result["checks"] if c["field"] == "expected_confluence_pct")
    assert conf["computed"] is not None


def test_culture_only_no_claimed_keys_all_not_claimed():
    result = verify_published_protocol(
        culture={"plate_format": "12-well", "seeding_density_cells_cm2": 5e4},
        claimed={},
        reference="10.1000/culture-noclaims",
    )
    assert result["status"] == "ok"
    assert all(c["verdict"] == "not_claimed" for c in result["checks"])


def test_culture_metrics_exports():
    assert CULTURE_METRICS == ("seed_per_well", "medium_volume_per_well_ml", "expected_confluence_pct")


def test_no_domain_is_validation_error():
    result = verify_published_protocol(
        chip={}, flow={}, claimed={}, reference="10.1000/empty"
    )
    assert result["status"] == "validation_error"


def test_missing_seeding_density_is_validation_error():
    result = verify_published_protocol(
        culture={"plate_format": "96-well"},  # no seeding density
        claimed={},
        reference="10.1000/culture-bad",
    )
    assert result["status"] == "validation_error"
