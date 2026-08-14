"""One declaration per design domain — the Block spec.

A **block** is one named, optional part of a design: the derived flow metrics,
the plate-culture plan, the 3D-spheroid plan, the channel cell plan, the dosing
plan, the statistics plan or the perfused-system PK plan. Each block is
declared **exactly once here**, and
everything the other layers need to know about it *declaratively* lives in that
one entry:

* ``raw_keys`` — the inputs the LLM may propose (nothing derived).
* ``derived_keys`` — the fields the calculators own; the design gate
  (:mod:`labwright.design`) refuses any of these from the model.
* ``consistency_keys`` — the minimal raw set a bare model must report for its
  numbers to be cross-checkable in the benchmark.
* ``field_map`` — how each report/gold key maps to a verifier field name
  (``"spheroid_volume_ul"`` → ``"spheroid.spheroid_volume_ul"``), which the
  benchmark's unit-misread layer looks up in the canonical-unit table.
* ``sanity_bands`` and ``canonical_units`` — the physiological range bands
  (:mod:`labwright.verify.sanity`) and the audit units
  (:mod:`labwright.verify.units`), keyed by verifier field name.

The consumers import their tables from this registry rather than carrying their
own per-domain constants, so adding a domain is: a ``calc/`` module + a derive
function + a schema model + one ``Block`` entry here. ``_validate()`` runs at
import time and raises if a declared derived field is missing its field-map
entry, sanity band or canonical unit — the "one declaration" property is itself
enforced, not just promised. ``tests/test_blocks.py`` pins the same contract.

This is the *design-domain* registry; :mod:`labwright.physiology` is the
companion *physiology-data* registry (cell-type / organ tables consumed by the
derivers, range checks and safety hints).

Data source for every value: ``labwright/verify/sanity.py`` (bands) and
``labwright/verify/units.py`` (units) before they were centralized here — the
numbers are unchanged, only their home moved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType

from labwright.calc import barrier as calc_barrier
from labwright.calc import breathing as calc_breathing
from labwright.calc import cell as calc_cell
from labwright.calc import culture as calc_culture
from labwright.calc import dosing as calc_dosing
from labwright.calc import gradient as calc_gradient
from labwright.calc import microfluidics as mf
from labwright.calc import o2 as calc_o2
from labwright.calc import pk as calc_pk
from labwright.calc import pulsatile as calc_pulsatile
from labwright.calc import pumpless as calc_pumpless
from labwright.calc import scaling as calc_scaling
from labwright.calc import spheroid as calc_spheroid
from labwright.calc import stats as calc_stats


@dataclass(frozen=True)
class Band:
    """A physiological range with a hard physical boundary.

    ``None`` bounds are unbounded on that side. ``soft_min/soft_max`` are the
    warning band; ``hard_min/hard_max`` the error band. Hard bounds are always
    wider than (or equal to) the soft bounds.

    Defined here (not in ``verify.sanity``) so a :class:`Block` can carry its
    bands without importing the verifier; ``verify.sanity`` re-exports it.
    """

    soft_min: float | None
    soft_max: float | None
    hard_min: float | None
    hard_max: float | None
    description: str
    units: str


@dataclass(frozen=True)
class Block:
    """The declarative metadata of one design domain."""

    name: str
    #: DesignPlan attribute holding the block's plan (``"derived"`` for flow).
    plan_field: str
    #: DesignInput attribute holding the raw inputs (``None`` for flow, whose
    #: raws span the separate ``chip`` and ``flow`` inputs).
    input_field: str | None
    #: The calculator module that owns this domain's arithmetic.
    calc: ModuleType
    raw_keys: tuple[str, ...]
    derived_keys: tuple[str, ...]
    consistency_keys: tuple[str, ...]
    #: report/gold key -> verifier field name (benchmark unit-misread layer).
    field_map: dict[str, str]
    #: verifier field name -> physiological range band.
    sanity_bands: dict[str, Band]
    #: verifier field name -> canonical unit (audit table).
    canonical_units: dict[str, str]


# ---------------------------------------------------------------------------
# The blocks, in registration order (flow → cells → culture → spheroid →
# dosing → stats → pk). Shared-key precedence in the merged tables follows this
# order: ``total_medium_ml`` is a derived key of both the culture and spheroid
# blocks (each with its own verifier field, both canonical unit "mL"), and the
# merged field map keeps the *first* mapping, i.e. culture's — matching the
# pre-registry ``_FIELD_MAP`` exactly. ``flow_rate_uLmin`` is a raw key of both
# the flow and pk blocks; the flow block's top-level mapping wins, so the
# merged field map resolves it to the flow field (canonical unit "uL/min"
# either way).
# ---------------------------------------------------------------------------


def _flow() -> Block:
    return Block(
        name="flow",
        plan_field="derived",
        input_field=None,
        calc=mf,
        raw_keys=(
            "width_um", "height_um", "length_mm", "flow_rate_uLmin",
            "viscosity_pas", "density_kgm3",
        ),
        derived_keys=(
            "shear_pa", "reynolds", "pressure_drop_pa", "residence_time_s",
            "channel_volume_ul", "mean_velocity_mms",
        ),
        consistency_keys=(
            "width_um", "height_um", "length_mm", "flow_rate_uLmin",
            "viscosity_pas", "density_kgm3",
        ),
        field_map={
            "shear_pa": "derived.shear_pa",
            "reynolds": "derived.reynolds",
            "pressure_drop_pa": "derived.pressure_drop_pa",
            "residence_time_s": "derived.residence_time_s",
            "channel_volume_ul": "derived.channel_volume_ul",
            "mean_velocity_mms": "derived.mean_velocity_mms",
            "flow_rate_uLmin": "flow_rate_uLmin",
        },
        sanity_bands={
            "derived.shear_pa": Band(0.001, 10, 1e-4, 50,
                "wall shear stress in organ-on-chip culture", "Pa"),
            "derived.reynolds": Band(0.001, 200, 0.0, 2300,
                "Reynolds number (flow must be laminar)", "dimensionless"),
            "derived.pressure_drop_pa": Band(1.0, 1e5, 0.0, 1e6,
                "laminar pressure drop along a microchannel", "Pa"),
            "derived.residence_time_s": Band(0.1, 1e4, 1e-4, 1e5,
                "fluid residence time in a channel", "s"),
            "derived.channel_volume_ul": Band(0.01, 100, 1e-6, 1e4,
                "per-channel culture volume", "uL"),
            "derived.mean_velocity_mms": Band(0.01, 1e4, 1e-4, 1e5,
                "mean flow velocity in a channel", "mm/s"),
        },
        canonical_units={
            "derived.shear_pa": "Pa",
            "derived.reynolds": "dimensionless",
            "derived.pressure_drop_pa": "Pa",
            "derived.residence_time_s": "s",
            "derived.channel_volume_ul": "uL",
            "derived.mean_velocity_mms": "mm/s",
            "flow_rate_uLmin": "uL/min",
            "viscosity_pas": "Pa*s",
            "density_kgm3": "kg/m^3",
            "width_um": "um",
            "height_um": "um",
            "length_mm": "mm",
        },
    )


def _cells() -> Block:
    return Block(
        name="cells",
        plan_field="cells",
        input_field="cells",
        calc=calc_cell,
        raw_keys=(
            "seeding_density_cells_cm2", "culture_area_cm2",
            "doubling_time_h", "culture_duration_h",
        ),
        derived_keys=("seed_count",),
        consistency_keys=(),
        field_map={"seed_count": "cells.seed_count"},
        sanity_bands={
            "cells.seed_count": Band(1e2, 1e9, 1.0, 1e10,
                "cells seeded onto a culture area", "cells"),
            "cells.seeding_density_cells_cm2": Band(1e3, 1e7, 1.0, 1e9,
                "cell seeding density", "cells/cm^2"),
            "cells.culture_area_cm2": Band(1e-4, 100, 1e-6, 1e3,
                "cell culture area", "cm^2"),
        },
        canonical_units={
            "cells.seed_count": "cells",
            "cells.seeding_density_cells_cm2": "cells/cm^2",
            "cells.culture_area_cm2": "cm^2",
        },
    )


def _culture() -> Block:
    return Block(
        name="culture",
        plan_field="culture",
        input_field="culture",
        calc=calc_culture,
        raw_keys=(
            "plate_format", "wells", "seeding_density_cells_cm2",
            "viability_pct", "confluent_density_cells_cm2", "doubling_time_h",
            "culture_duration_h",
        ),
        derived_keys=(
            "seed_per_well", "total_seed_count", "medium_volume_per_well_ml",
            "total_medium_ml", "expected_confluence_pct",
        ),
        consistency_keys=("plate_format", "seeding_density_cells_cm2", "wells"),
        field_map={
            "seed_per_well": "culture.seed_per_well",
            "total_seed_count": "culture.total_seed_count",
            "medium_volume_per_well_ml": "culture.medium_volume_per_well_ml",
            "total_medium_ml": "culture.total_medium_ml",
            "expected_confluence_pct": "culture.expected_confluence_pct",
        },
        sanity_bands={
            "culture.seed_per_well": Band(1e2, 1e9, 1.0, 1e10,
                "cells seeded per well", "cells"),
            "culture.total_seed_count": Band(1e2, 1e10, 1.0, 1e11,
                "total cells seeded across wells", "cells"),
            "culture.seeding_density_cells_cm2": Band(1e3, 1e7, 1.0, 1e9,
                "cell seeding density", "cells/cm^2"),
            "culture.medium_volume_per_well_ml": Band(0.01, 5.0, 1e-4, 100,
                "working medium volume per well", "mL"),
            "culture.total_medium_ml": Band(0.01, 1e3, 1e-4, 1e4,
                "total medium volume across wells", "mL"),
            "culture.expected_confluence_pct": Band(0.0, 100, 0.0, 1000,
                "predicted confluence at harvest (may exceed 100 % for over-confluent cultures)", "%"),
            "culture.doubling_time_h": Band(10, 200, 0.1, 1000,
                "population doubling time", "h"),
            "culture.culture_duration_h": Band(0.0, 2000, 0.0, 1e5,
                "culture duration", "h"),
        },
        canonical_units={
            "culture.seed_per_well": "cells",
            "culture.total_seed_count": "cells",
            "culture.medium_volume_per_well_ml": "mL",
            "culture.total_medium_ml": "mL",
            "culture.seeding_density_cells_cm2": "cells/cm^2",
            "culture.expected_confluence_pct": "%",
            "culture.viability_pct": "%",
            "culture.doubling_time_h": "h",
            "culture.culture_duration_h": "h",
        },
    )


def _spheroid() -> Block:
    return Block(
        name="spheroid",
        plan_field="spheroid",
        input_field="spheroid",
        calc=calc_spheroid,
        raw_keys=(
            "spheroid_format", "spheroid_count", "cells_per_spheroid",
            "cell_diameter_um", "doubling_time_h", "culture_duration_h",
        ),
        derived_keys=(
            "spheroid_volume_ul", "expected_diameter_um", "cells_total",
            "medium_volume_per_spheroid_ul", "total_medium_ml",
            "expected_cells_after_growth",
        ),
        consistency_keys=(
            "spheroid_format", "spheroid_count", "cells_per_spheroid",
            "cell_diameter_um",
        ),
        field_map={
            "spheroid_volume_ul": "spheroid.spheroid_volume_ul",
            "expected_diameter_um": "spheroid.expected_diameter_um",
            "cells_total": "spheroid.cells_total",
            "medium_volume_per_spheroid_ul": "spheroid.medium_volume_per_spheroid_ul",
            "total_medium_ml": "spheroid.total_medium_ml",
            "cells_per_spheroid": "spheroid.cells_per_spheroid",
            "spheroid_count": "spheroid.spheroid_count",
            "expected_cells_after_growth": "spheroid.expected_cells_after_growth",
        },
        sanity_bands={
            "spheroid.cells_per_spheroid": Band(100, 1e5, 1.0, 1e6,
                "cells seeded per spheroid", "cells"),
            "spheroid.expected_diameter_um": Band(30, 2000, 5, 1e4,
                "spheroid diameter (functional spheroids stay < ~400 µm to avoid necrotic cores)", "um"),
            "spheroid.spheroid_volume_ul": Band(1e-4, 1e2, 1e-6, 1e4,
                "spheroid volume (a 200 µm spheroid ≈ 4.2e-3 uL)", "uL"),
            "spheroid.cell_diameter_um": Band(5, 60, 1.0, 200,
                "mean single-cell diameter", "um"),
            "spheroid.medium_volume_per_spheroid_ul": Band(10, 300, 1.0, 2000,
                "working medium volume per spheroid", "uL"),
            "spheroid.total_medium_ml": Band(0.01, 1e3, 1e-4, 1e4,
                "total medium volume for the spheroid culture", "mL"),
            "spheroid.cells_total": Band(1e2, 1e10, 1.0, 1e11,
                "total cells for spheroid seeding", "cells"),
            "spheroid.spheroid_count": Band(1, 1e5, 1.0, 1e6,
                "number of spheroids", "n"),
            "spheroid.doubling_time_h": Band(10, 200, 0.1, 1000,
                "population doubling time", "h"),
            "spheroid.culture_duration_h": Band(0.0, 2000, 0.0, 1e5,
                "culture duration", "h"),
            "spheroid.expected_cells_after_growth": Band(10, 1e8, 1.0, 1e10,
                "predicted cells per spheroid at harvest", "cells"),
        },
        canonical_units={
            "spheroid.cells_per_spheroid": "cells",
            "spheroid.spheroid_count": "n",
            "spheroid.expected_diameter_um": "um",
            "spheroid.spheroid_volume_ul": "uL",
            "spheroid.cell_diameter_um": "um",
            "spheroid.medium_volume_per_spheroid_ul": "uL",
            "spheroid.total_medium_ml": "mL",
            "spheroid.cells_total": "cells",
            "spheroid.expected_cells_after_growth": "cells",
        },
    )


def _dosing() -> Block:
    return Block(
        name="dosing",
        plan_field="dosing",
        input_field="dosing",
        calc=calc_dosing,
        raw_keys=(
            "compound", "molecular_weight_g_mol", "stock_mM", "working_mM",
            "vehicle_control", "exposure_h",
        ),
        derived_keys=("dmso_fraction_vv",),
        consistency_keys=(),
        field_map={"dmso_fraction_vv": "dosing.dmso_fraction_vv"},
        sanity_bands={
            "dosing.stock_mM": Band(0.1, 1e4, 1e-4, 1e6,
                "compound stock concentration", "mM"),
            "dosing.working_mM": Band(1e-3, 100, 1e-6, 1e4,
                "compound working concentration", "mM"),
            "dosing.dmso_fraction_vv": Band(0.0, 0.005, 0.0, 0.14,
                "DMSO volume fraction in medium", "v/v"),
        },
        canonical_units={
            "dosing.stock_mM": "mM",
            "dosing.working_mM": "mM",
            "dosing.dmso_fraction_vv": "v/v",
            "dosing.molecular_weight_g_mol": "g/mol",
        },
    )


def _stats() -> Block:
    return Block(
        name="stats",
        plan_field="stats",
        input_field="stats",
        calc=calc_stats,
        raw_keys=("effect_size", "std_dev", "alpha", "power"),
        derived_keys=("n_per_group",),
        consistency_keys=(),
        field_map={"n_per_group": "stats.n_per_group"},
        sanity_bands={
            "stats.n_per_group": Band(3, 1000, 2, 1e6,
                "biological replicates per group", "n"),
        },
        canonical_units={"stats.n_per_group": "n"},
    )


def _pk() -> Block:
    """Perfused-system pharmacokinetics block.

    Raw inputs are the measured inlet/outlet concentrations and the perfusion
    flow; the calculators own extraction ratio, clearance and — when the extra
    inputs are present — half-life, accumulation ratio and mass cleared. The
    ``flow_rate_uLmin`` here is the PK circuit's own field (declared inside the
    ``pk`` plan), independent of the flow block's top-level field — the merged
    field map keeps the flow block's first-wins mapping.
    """
    return Block(
        name="pk",
        plan_field="pk",
        input_field="pk",
        calc=calc_pk,
        raw_keys=(
            "compound", "inlet_concentration_uM", "outlet_concentration_uM",
            "flow_rate_uLmin", "system_volume_uL", "dose_interval_h",
            "molecular_weight_g_mol",
        ),
        derived_keys=(
            "extraction_ratio", "clearance_uLmin", "half_life_h",
            "accumulation_ratio", "mass_cleared_ug_h",
        ),
        consistency_keys=(
            "inlet_concentration_uM", "outlet_concentration_uM", "flow_rate_uLmin",
        ),
        field_map={
            "extraction_ratio": "pk.extraction_ratio",
            "clearance_uLmin": "pk.clearance_uLmin",
            "half_life_h": "pk.half_life_h",
            "accumulation_ratio": "pk.accumulation_ratio",
            "mass_cleared_ug_h": "pk.mass_cleared_ug_h",
            "inlet_concentration_uM": "pk.inlet_concentration_uM",
            "outlet_concentration_uM": "pk.outlet_concentration_uM",
            "flow_rate_uLmin": "pk.flow_rate_uLmin",
            "system_volume_uL": "pk.system_volume_uL",
            "dose_interval_h": "pk.dose_interval_h",
            "molecular_weight_g_mol": "pk.molecular_weight_g_mol",
        },
        sanity_bands={
            "pk.extraction_ratio": Band(0.0, 0.99, -1.0, 1.0,
                "fraction of drug extracted in one pass (negative = net secretion)", "dimensionless"),
            "pk.clearance_uLmin": Band(0.01, 1e3, -1e3, 1e5,
                "volume of perfusate cleared of drug per minute", "uL/min"),
            "pk.half_life_h": Band(0.01, 200, 1e-4, 1e4,
                "elimination half-life in a recirculating OOC system", "h"),
            "pk.accumulation_ratio": Band(1.0, 100, 1.0, 1e6,
                "steady-state accumulation factor under repeated dosing", "dimensionless"),
            "pk.mass_cleared_ug_h": Band(1e-6, 1e3, 0.0, 1e9,
                "mass of drug the chip clears per hour", "ug/h"),
            "pk.inlet_concentration_uM": Band(1e-3, 1e3, 1e-6, 1e6,
                "drug concentration entering the chip", "uM"),
            "pk.outlet_concentration_uM": Band(0.0, 1e3, 0.0, 1e6,
                "drug concentration leaving the chip", "uM"),
            "pk.flow_rate_uLmin": Band(0.1, 1e3, 1e-3, 1e5,
                "perfusion flow rate in the PK circuit", "uL/min"),
            "pk.system_volume_uL": Band(50, 1e5, 1.0, 1e7,
                "recirculating medium volume (reservoir + chip + tubing)", "uL"),
            "pk.dose_interval_h": Band(0.5, 168, 1e-3, 1e5,
                "time between doses", "h"),
            "pk.molecular_weight_g_mol": Band(100, 1e3, 10, 1e5,
                "drug molecular weight", "g/mol"),
        },
        canonical_units={
            "pk.extraction_ratio": "dimensionless",
            "pk.clearance_uLmin": "uL/min",
            "pk.half_life_h": "h",
            "pk.accumulation_ratio": "dimensionless",
            "pk.mass_cleared_ug_h": "ug/h",
            "pk.inlet_concentration_uM": "uM",
            "pk.outlet_concentration_uM": "uM",
            "pk.flow_rate_uLmin": "uL/min",
            "pk.system_volume_uL": "uL",
            "pk.dose_interval_h": "h",
            "pk.molecular_weight_g_mol": "g/mol",
        },
    )


def _barrier() -> Block:
    """Epithelial/endothelial barrier QC block.

    Raw inputs are the measured resistances (plus the probe flux/conc when a
    permeability readout is planned); the calculators own TEER, Papp and the
    permeability-surface-area product (clearance).
    """
    return Block(
        name="barrier",
        plan_field="barrier",
        input_field="barrier",
        calc=calc_barrier,
        raw_keys=(
            "cell_type", "insert_area_cm2", "resistance_total_ohm",
            "resistance_blank_ohm", "probe", "donor_conc_um", "flux_nmol_min",
        ),
        derived_keys=("teer_ohm_cm2", "papp_cm_s", "clearance_mL_min"),
        consistency_keys=("insert_area_cm2", "resistance_total_ohm", "resistance_blank_ohm"),
        field_map={
            "teer_ohm_cm2": "barrier.teer_ohm_cm2",
            "papp_cm_s": "barrier.papp_cm_s",
            "clearance_mL_min": "barrier.clearance_mL_min",
            "insert_area_cm2": "barrier.insert_area_cm2",
            "resistance_total_ohm": "barrier.resistance_total_ohm",
            "resistance_blank_ohm": "barrier.resistance_blank_ohm",
            "donor_conc_um": "barrier.donor_conc_um",
            "flux_nmol_min": "barrier.flux_nmol_min",
        },
        sanity_bands={
            "barrier.teer_ohm_cm2": Band(50, 2500, 1.0, 1e5,
                "monolayer TEER (BBB in vivo ~1500-2000, Caco-2 ~921, leaky epithelia ~50)", "ohm*cm^2"),
            "barrier.papp_cm_s": Band(1e-7, 1e-5, 1e-9, 1e-4,
                "apparent permeability (tight barriers ~1e-7-1e-6 cm/s)", "cm/s"),
            "barrier.clearance_mL_min": Band(1e-5, 1.0, 1e-8, 1e3,
                "permeability-surface-area product", "mL/min"),
            "barrier.insert_area_cm2": Band(0.01, 10, 1e-4, 100,
                "membrane growth area", "cm^2"),
            "barrier.resistance_total_ohm": Band(50, 1e5, 1.0, 1e6,
                "total insert resistance", "ohm"),
            "barrier.resistance_blank_ohm": Band(1.0, 1e4, 0.1, 1e5,
                "cell-free insert resistance", "ohm"),
            "barrier.donor_conc_um": Band(0.1, 1e3, 1e-3, 1e5,
                "donor probe concentration", "uM"),
            "barrier.flux_nmol_min": Band(1e-3, 1e3, 0.0, 1e6,
                "steady-state probe flux", "nmol/min"),
        },
        canonical_units={
            "barrier.teer_ohm_cm2": "ohm*cm^2",
            "barrier.papp_cm_s": "cm/s",
            "barrier.clearance_mL_min": "mL/min",
            "barrier.insert_area_cm2": "cm^2",
            "barrier.resistance_total_ohm": "ohm",
            "barrier.resistance_blank_ohm": "ohm",
            "barrier.donor_conc_um": "uM",
            "barrier.flux_nmol_min": "nmol/min",
        },
    )


def _oxygen() -> Block:
    """Dissolved-oxygen control block.

    Raw inputs are the target pO2 and the consumption-relevant cell density /
    spheroid diameter; the calculators own dissolved concentration (Henry),
    Krogh penetration depth and — for spheroids — the necrotic-core fraction.
    The cell-type OCR comes from :mod:`labwright.physiology`, not the LLM.
    """
    return Block(
        name="oxygen",
        plan_field="oxygen",
        input_field="oxygen",
        calc=calc_o2,
        raw_keys=(
            "cell_type", "target_po2_mmhg", "cell_density_cells_ml",
            "spheroid_diameter_um",
        ),
        derived_keys=(
            "dissolved_o2_mM", "penetration_depth_um", "necrotic_fraction",
            "demand_umol_min",
        ),
        consistency_keys=("target_po2_mmhg",),
        field_map={
            "dissolved_o2_mM": "oxygen.dissolved_o2_mM",
            "penetration_depth_um": "oxygen.penetration_depth_um",
            "necrotic_fraction": "oxygen.necrotic_fraction",
            "demand_umol_min": "oxygen.demand_umol_min",
            "target_po2_mmhg": "oxygen.target_po2_mmhg",
        },
        sanity_bands={
            "oxygen.target_po2_mmhg": Band(5, 160, 0.0, 760,
                "tissue O2 partial pressure (in vivo 8-104 mmHg; air-equilibrated medium ~150)", "mmHg"),
            "oxygen.dissolved_o2_mM": Band(0.005, 0.3, 0.0, 1.0,
                "dissolved O2 in medium (air-saturated ~0.2 mM)", "mM"),
            "oxygen.penetration_depth_um": Band(10, 400, 1.0, 1e4,
                "Krogh O2 penetration depth into consuming tissue", "um"),
            "oxygen.necrotic_fraction": Band(0.0, 0.9, 0.0, 1.0,
                "spheroid anoxic-core volume fraction", "dimensionless"),
            "oxygen.demand_umol_min": Band(1e-3, 100, 0.0, 1e6,
                "O2 demand per 1e6 cells at registry OCR", "umol/min"),
        },
        canonical_units={
            "oxygen.target_po2_mmhg": "mmHg",
            "oxygen.dissolved_o2_mM": "mM",
            "oxygen.penetration_depth_um": "um",
            "oxygen.necrotic_fraction": "dimensionless",
            "oxygen.demand_umol_min": "umol/min",
        },
    )


def _pumpless() -> Block:
    """Gravity-driven pumpless (rocking/tilting) perfusion block.

    Raw inputs are the platform settings (tilt, rocking half-period) and the
    channel geometry; the calculators own the hydrostatic head, the driven flow
    rate, the peak wall shear, the volume displaced per half-cycle, the
    oscillatory shear index and the cycles-per-hour. The physiological shear
    target comes from the cell-type registry (falling back to the liver
    sinusoidal range cited for gravity-driven chips), never the LLM.
    """
    return Block(
        name="pumpless",
        plan_field="pumpless",
        input_field="pumpless",
        calc=calc_pumpless,
        raw_keys=(
            "cell_type", "tilt_angle_deg", "channel_length_mm", "width_um",
            "height_um", "rocking_half_period_s", "viscosity_pas",
            "density_kgm3", "backward_shear_fraction",
        ),
        derived_keys=(
            "hydrostatic_head_pa", "driven_flow_rate_uLmin", "peak_wall_shear_pa",
            "volume_per_half_cycle_ul", "oscillatory_shear_index",
            "cycles_per_hour", "shear_ratio_to_target",
        ),
        consistency_keys=(
            "tilt_angle_deg", "channel_length_mm", "width_um", "height_um",
            "rocking_half_period_s",
        ),
        field_map={
            "hydrostatic_head_pa": "pumpless.hydrostatic_head_pa",
            "driven_flow_rate_uLmin": "pumpless.driven_flow_rate_uLmin",
            "peak_wall_shear_pa": "pumpless.peak_wall_shear_pa",
            "volume_per_half_cycle_ul": "pumpless.volume_per_half_cycle_ul",
            "oscillatory_shear_index": "pumpless.oscillatory_shear_index",
            "cycles_per_hour": "pumpless.cycles_per_hour",
            "shear_ratio_to_target": "pumpless.shear_ratio_to_target",
            "tilt_angle_deg": "pumpless.tilt_angle_deg",
            "channel_length_mm": "pumpless.channel_length_mm",
            "rocking_half_period_s": "pumpless.rocking_half_period_s",
        },
        sanity_bands={
            "pumpless.tilt_angle_deg": Band(1, 25, 0.0, 45,
                "platform tilt from horizontal (MIMETAS rocker limit 25)", "deg"),
            "pumpless.channel_length_mm": Band(5, 100, 0.5, 500,
                "channel length along the tilt axis", "mm"),
            "pumpless.rocking_half_period_s": Band(5, 60, 1.0, 600,
                "rocking half-period (organ chips 5-60 s)", "s"),
            "pumpless.hydrostatic_head_pa": Band(0.1, 1e3, 1e-3, 1e4,
                "hydrostatic pressure head from platform tilt", "Pa"),
            "pumpless.driven_flow_rate_uLmin": Band(0.1, 1e3, 1e-3, 1e5,
                "gravity-driven flow rate", "uL/min"),
            "pumpless.peak_wall_shear_pa": Band(0.001, 1.0, 1e-5, 10,
                "peak wall shear during a rocking half-cycle (liver sinusoid 0.01-0.05 Pa)", "Pa"),
            "pumpless.volume_per_half_cycle_ul": Band(0.1, 1e3, 1e-3, 1e5,
                "volume displaced per rocking half-cycle", "uL"),
            "pumpless.oscillatory_shear_index": Band(0.0, 0.5, 0.0, 0.5,
                "oscillatory shear index (0 unidirectional, 0.5 symmetric)", "dimensionless"),
            "pumpless.cycles_per_hour": Band(1, 360, 0.1, 1e4,
                "rocking cycles per hour", "1/h"),
            "pumpless.shear_ratio_to_target": Band(0.5, 2.0, 0.05, 20,
                "chip WSS / physiological target (0.5-2x in range)", "dimensionless"),
        },
        canonical_units={
            "pumpless.hydrostatic_head_pa": "Pa",
            "pumpless.driven_flow_rate_uLmin": "uL/min",
            "pumpless.peak_wall_shear_pa": "Pa",
            "pumpless.volume_per_half_cycle_ul": "uL",
            "pumpless.oscillatory_shear_index": "dimensionless",
            "pumpless.cycles_per_hour": "1/h",
            "pumpless.shear_ratio_to_target": "dimensionless",
            "pumpless.tilt_angle_deg": "deg",
            "pumpless.channel_length_mm": "mm",
            "pumpless.rocking_half_period_s": "s",
        },
    )


def _breathing() -> Block:
    """Lung-on-chip ALI + cyclic mechanical stretch block.

    Raw inputs are the breathing frequency, applied strain and the optional
    ALI-film / stretch-duty / duration inputs; the calculators own breaths/min,
    membrane stroke, strain rate, total cycles, duty fraction and the apical
    liquid-film thickness. The physiological strain window (5-12 %, >20 %
    pathological) is a checker cross-check, not a raw choice.
    """
    return Block(
        name="breathing",
        plan_field="breathing",
        input_field="breathing",
        calc=calc_breathing,
        raw_keys=(
            "cell_type", "frequency_hz", "strain_pct", "membrane_span_um",
            "apical_volume_ul", "surface_area_cm2", "culture_duration_h",
            "stretch_seconds", "cycle_seconds",
        ),
        derived_keys=(
            "breaths_per_minute", "cyclic_displacement_um", "strain_rate_per_s",
            "total_cycles", "stretch_duty_fraction", "ali_liquid_film_um",
        ),
        consistency_keys=("frequency_hz", "strain_pct"),
        field_map={
            "breaths_per_minute": "breathing.breaths_per_minute",
            "cyclic_displacement_um": "breathing.cyclic_displacement_um",
            "strain_rate_per_s": "breathing.strain_rate_per_s",
            "total_cycles": "breathing.total_cycles",
            "stretch_duty_fraction": "breathing.stretch_duty_fraction",
            "ali_liquid_film_um": "breathing.ali_liquid_film_um",
            "frequency_hz": "breathing.frequency_hz",
            "strain_pct": "breathing.strain_pct",
            "membrane_span_um": "breathing.membrane_span_um",
        },
        sanity_bands={
            "breathing.frequency_hz": Band(0.2, 0.25, 0.01, 2.0,
                "breathing frequency (physiological ~0.2-0.25 Hz)", "Hz"),
            "breathing.strain_pct": Band(5.0, 12.0, 0.0, 50,
                "alveolar linear strain (5-12% physiological, >20% pathological)", "%"),
            "breathing.membrane_span_um": Band(100, 500, 10, 5000,
                "alveolar membrane span", "um"),
            "breathing.breaths_per_minute": Band(12, 15, 0.6, 120,
                "respiratory rate", "breaths/min"),
            "breathing.cyclic_displacement_um": Band(1, 100, 0.1, 1000,
                "membrane edge stroke for the target strain", "um"),
            "breathing.strain_rate_per_s": Band(0.01, 0.05, 0.001, 1.0,
                "linearised strain rate (10% at 0.2 Hz -> 0.02 /s)", "1/s"),
            "breathing.total_cycles": Band(1, 1e6, 1.0, 1e9,
                "total stretch cycles over the culture", "n"),
            "breathing.stretch_duty_fraction": Band(0.0, 1.0, 0.0, 1.0,
                "fraction of each cycle held stretched", "dimensionless"),
            "breathing.ali_liquid_film_um": Band(1, 1000, 0.1, 1e4,
                "apical liquid film thickness at ALI", "um"),
        },
        canonical_units={
            "breathing.breaths_per_minute": "breaths/min",
            "breathing.cyclic_displacement_um": "um",
            "breathing.strain_rate_per_s": "1/s",
            "breathing.total_cycles": "n",
            "breathing.stretch_duty_fraction": "dimensionless",
            "breathing.ali_liquid_film_um": "um",
            "breathing.frequency_hz": "Hz",
            "breathing.strain_pct": "%",
            "breathing.membrane_span_um": "um",
        },
    )


def _pulsatile() -> Block:
    """Pulsatile cardiac-waveform (heart-on-chip) block.

    Raw inputs are the cardiac-cycle frequency, channel height and the
    sinusoidal shear mean/amplitude (plus the optional flow-waveform
    peak/min/mean); the calculators own the Womersley number, OSI, peak shear
    and the Gosling pulsatility index. The shear waveform shape is what the
    cells transduce — this block checks the *shape*, not just the mean.
    """
    return Block(
        name="pulsatile",
        plan_field="pulsatile",
        input_field="pulsatile",
        calc=calc_pulsatile,
        raw_keys=(
            "cell_type", "frequency_hz", "channel_height_um", "viscosity_pas",
            "density_kgm3", "shear_mean_pa", "shear_amplitude_pa",
            "peak_flow_uLmin", "minimum_flow_uLmin", "mean_flow_uLmin",
        ),
        derived_keys=(
            "womersley_number", "oscillatory_shear_index", "peak_shear_pa",
            "pulsatility_index",
        ),
        consistency_keys=(
            "frequency_hz", "channel_height_um", "shear_mean_pa",
            "shear_amplitude_pa",
        ),
        field_map={
            "womersley_number": "pulsatile.womersley_number",
            "oscillatory_shear_index": "pulsatile.oscillatory_shear_index",
            "peak_shear_pa": "pulsatile.peak_shear_pa",
            "pulsatility_index": "pulsatile.pulsatility_index",
            "frequency_hz": "pulsatile.frequency_hz",
            "channel_height_um": "pulsatile.channel_height_um",
            "shear_mean_pa": "pulsatile.shear_mean_pa",
            "shear_amplitude_pa": "pulsatile.shear_amplitude_pa",
        },
        sanity_bands={
            "pulsatile.frequency_hz": Band(0.8, 2.0, 0.1, 5.0,
                "cardiac-cycle frequency (48-120 bpm)", "Hz"),
            "pulsatile.channel_height_um": Band(50, 300, 5, 2000,
                "channel height", "um"),
            "pulsatile.womersley_number": Band(0.1, 2.0, 0.01, 10,
                "Womersley number (heart-on-chip demo ~0.27)", "dimensionless"),
            "pulsatile.oscillatory_shear_index": Band(0.0, 0.5, 0.0, 0.5,
                "oscillatory shear index (reversal fraction)", "dimensionless"),
            "pulsatile.peak_shear_pa": Band(0.001, 10, 1e-5, 100,
                "peak wall shear of the waveform (demo aortic inflow 0.59 Pa)", "Pa"),
            "pulsatile.pulsatility_index": Band(0.5, 20, 0.0, 1e3,
                "Gosling pulsatility index (arterial waveforms PI >= 1)", "dimensionless"),
            "pulsatile.shear_mean_pa": Band(0.001, 10, 1e-5, 100,
                "time-averaged wall shear", "Pa"),
            "pulsatile.shear_amplitude_pa": Band(0.0, 10, 0.0, 100,
                "shear oscillation amplitude", "Pa"),
        },
        canonical_units={
            "pulsatile.womersley_number": "dimensionless",
            "pulsatile.oscillatory_shear_index": "dimensionless",
            "pulsatile.peak_shear_pa": "Pa",
            "pulsatile.pulsatility_index": "dimensionless",
            "pulsatile.frequency_hz": "Hz",
            "pulsatile.channel_height_um": "um",
            "pulsatile.shear_mean_pa": "Pa",
            "pulsatile.shear_amplitude_pa": "Pa",
        },
    )


def _scaling() -> Block:
    """Multi-organ body-on-chip allometric scaling block.

    Raw inputs are the organ name, the chip-wide cell budget and the scaled
    cardiac output (plus optional compartment volume/flow for transit matching);
    the calculators own the organ flow fraction, organ perfusion flow, the
    mass-proportional cell number, the Kleiber allometric factor and the transit
    / residence-match numbers. Organ mass and flow fractions are physiology-table
    pinned (:mod:`labwright.calc.scaling`), never proposed by the LLM.
    """
    return Block(
        name="scaling",
        plan_field="scaling",
        input_field="scaling",
        calc=calc_scaling,
        raw_keys=(
            "organ", "total_cells_chip", "cardiac_output_mlmin", "body_mass_g",
            "chip_volume_ul", "flow_rate_uLmin", "target_transit_s",
        ),
        derived_keys=(
            "organ_flow_fraction", "organ_flow_rate_mlmin", "cells_in_organ",
            "allometric_scale", "transit_time_s", "residence_time_match_error_s",
        ),
        consistency_keys=("organ", "total_cells_chip", "cardiac_output_mlmin"),
        field_map={
            "organ_flow_fraction": "scaling.organ_flow_fraction",
            "organ_flow_rate_mlmin": "scaling.organ_flow_rate_mlmin",
            "cells_in_organ": "scaling.cells_in_organ",
            "allometric_scale": "scaling.allometric_scale",
            "transit_time_s": "scaling.transit_time_s",
            "residence_time_match_error_s": "scaling.residence_time_match_error_s",
            "total_cells_chip": "scaling.total_cells_chip",
            "cardiac_output_mlmin": "scaling.cardiac_output_mlmin",
        },
        sanity_bands={
            "scaling.organ_flow_fraction": Band(0.05, 0.27, 0.01, 0.5,
                "fraction of cardiac output perfusing the organ", "dimensionless"),
            "scaling.organ_flow_rate_mlmin": Band(10, 2000, 1.0, 1e5,
                "organ perfusion flow (adult cardiac output 5000 mL/min)", "mL/min"),
            "scaling.cells_in_organ": Band(1e3, 1e9, 1.0, 1e12,
                "cells assigned to the organ compartment (mass-proportional)", "cells"),
            "scaling.allometric_scale": Band(0.01, 0.5, 1e-4, 1.0,
                "allometric metabolic scale factor (mass^0.75)", "dimensionless"),
            "scaling.transit_time_s": Band(1, 1e4, 0.1, 1e6,
                "perfusate transit time through the compartment", "s"),
            "scaling.residence_time_match_error_s": Band(0, 3600, 0.0, 1e6,
                "absolute transit-time mismatch vs in-vivo target", "s"),
            "scaling.total_cells_chip": Band(1e4, 1e9, 1e2, 1e12,
                "chip-wide cell budget", "cells"),
            "scaling.cardiac_output_mlmin": Band(1, 5000, 0.1, 1e5,
                "cardiac output the chip is scaled to", "mL/min"),
        },
        canonical_units={
            "scaling.organ_flow_fraction": "dimensionless",
            "scaling.organ_flow_rate_mlmin": "mL/min",
            "scaling.cells_in_organ": "cells",
            "scaling.allometric_scale": "dimensionless",
            "scaling.transit_time_s": "s",
            "scaling.residence_time_match_error_s": "s",
            "scaling.total_cells_chip": "cells",
            "scaling.cardiac_output_mlmin": "mL/min",
        },
    )


def _gradient() -> Block:
    """Steady concentration-gradient (chemotaxis) block.

    Raw inputs are the source/sink concentrations, the source-sink gap and the
    experiment duration; the calculators own the gradient steepness, the
    mid-gap concentration, the diffusive relaxation time and the steady-state
    flux. Gradient stability (experiment >= 10τ) is a checker cross-check.
    """
    return Block(
        name="gradient",
        plan_field="gradient",
        input_field="gradient",
        calc=calc_gradient,
        raw_keys=(
            "chemoattractant", "source_conc_um", "sink_conc_um", "distance_um",
            "experiment_hours", "diffusivity_m2s",
        ),
        derived_keys=(
            "steepness_um_per_mm", "midpoint_conc_um", "relaxation_time_s",
            "flux_mol_m2s",
        ),
        consistency_keys=(
            "source_conc_um", "sink_conc_um", "distance_um",
        ),
        field_map={
            "steepness_um_per_mm": "gradient.steepness_um_per_mm",
            "midpoint_conc_um": "gradient.midpoint_conc_um",
            "relaxation_time_s": "gradient.relaxation_time_s",
            "flux_mol_m2s": "gradient.flux_mol_m2s",
            "source_conc_um": "gradient.source_conc_um",
            "sink_conc_um": "gradient.sink_conc_um",
            "distance_um": "gradient.distance_um",
            "experiment_hours": "gradient.experiment_hours",
        },
        sanity_bands={
            "gradient.source_conc_um": Band(0.01, 1e4, 1e-4, 1e6,
                "source chemoattractant concentration", "uM"),
            "gradient.sink_conc_um": Band(0.0, 1e4, 0.0, 1e6,
                "sink buffer concentration", "uM"),
            "gradient.distance_um": Band(100, 5000, 10, 1e5,
                "source-sink gap (classic ~1000 um)", "um"),
            "gradient.steepness_um_per_mm": Band(0.1, 1e4, 0.01, 1e6,
                "gradient steepness (a 90 uM/mm CXCL12 gradient is typical)", "uM/mm"),
            "gradient.midpoint_conc_um": Band(0.0, 1e4, 0.0, 1e6,
                "mid-gap steady-state concentration", "uM"),
            "gradient.relaxation_time_s": Band(1, 1e5, 0.1, 1e7,
                "diffusive gradient formation time (1 mm agarose ~30 min)", "s"),
            "gradient.flux_mol_m2s": Band(1e-10, 1e-4, 1e-14, 1.0,
                "steady-state diffusive flux", "mol/m^2/s"),
            "gradient.experiment_hours": Band(0.5, 168, 0.01, 1e5,
                "experiment duration", "h"),
        },
        canonical_units={
            "gradient.steepness_um_per_mm": "uM/mm",
            "gradient.midpoint_conc_um": "uM",
            "gradient.relaxation_time_s": "s",
            "gradient.flux_mol_m2s": "mol/m^2/s",
            "gradient.source_conc_um": "uM",
            "gradient.sink_conc_um": "uM",
            "gradient.distance_um": "um",
            "gradient.experiment_hours": "h",
        },
    )


BLOCKS: dict[str, Block] = {
    b.name: b
    for b in (
        _flow(), _cells(), _culture(), _spheroid(), _dosing(), _stats(),
        _pk(), _barrier(), _oxygen(),
        _pumpless(), _breathing(), _pulsatile(), _scaling(), _gradient(),
    )
}


# ---------------------------------------------------------------------------
# Merged tables the consumers import (single source of truth).
# ---------------------------------------------------------------------------


def _merge() -> dict[str, str]:
    """Union of all blocks' field maps, first declaration wins.

    ``total_medium_ml`` is a derived key of both the culture and spheroid
    blocks (each mapped to its own verifier field, both canonical unit "mL");
    culture is declared first, so its mapping wins — matching the pre-registry
    ``_FIELD_MAP`` entry exactly.
    """
    out: dict[str, str] = {}
    for b in BLOCKS.values():
        for key, field_name in b.field_map.items():
            if key not in out:
                out[key] = field_name
    return out


ALL_FIELD_MAP: dict[str, str] = _merge()

ALL_DERIVED_KEYS: frozenset[str] = frozenset(
    key for b in BLOCKS.values() for key in b.derived_keys
)

ALL_RAW_KEYS: frozenset[str] = frozenset(
    key for b in BLOCKS.values() for key in b.raw_keys
)

ALL_SANITY_BANDS: dict[str, Band] = {
    field_name: band
    for b in BLOCKS.values()
    for field_name, band in b.sanity_bands.items()
}

ALL_CANONICAL_UNITS: dict[str, str] = {
    field_name: unit
    for b in BLOCKS.values()
    for field_name, unit in b.canonical_units.items()
}


def _validate() -> None:
    """Fail fast if a block's declaration is incomplete.

    Every declared derived field must have a field-map entry, a sanity band and
    a canonical unit; every field-map value must resolve to a canonical unit;
    no key may be both raw and derived in the same block. This runs at import
    time, so a future domain that forgets a band or unit breaks loudly instead
    of silently scoring.
    """
    for name, b in BLOCKS.items():
        dupes = set(b.raw_keys) & set(b.derived_keys)
        if dupes:
            raise ValueError(f"block {name!r}: keys are both raw and derived: {sorted(dupes)}")
        for dk in b.derived_keys:
            field_name = b.field_map.get(dk)
            if field_name is None:
                raise ValueError(f"block {name!r}: derived key {dk!r} has no field_map entry")
            if field_name not in b.sanity_bands:
                raise ValueError(
                    f"block {name!r}: derived field {field_name!r} has no sanity band"
                )
            if field_name not in b.canonical_units:
                raise ValueError(
                    f"block {name!r}: derived field {field_name!r} has no canonical unit"
                )
        for key, field_name in b.field_map.items():
            if field_name != key and not field_name.startswith(b.plan_field + "."):
                raise ValueError(
                    f"block {name!r}: field_map value {field_name!r} does not match "
                    f"its block prefix {b.plan_field + '.'!r}"
                )
            if field_name not in b.canonical_units:
                raise ValueError(
                    f"block {name!r}: field_map value {field_name!r} has no canonical unit"
                )


_validate()


__all__ = [
    "Band",
    "Block",
    "BLOCKS",
    "ALL_FIELD_MAP",
    "ALL_DERIVED_KEYS",
    "ALL_RAW_KEYS",
    "ALL_SANITY_BANDS",
    "ALL_CANONICAL_UNITS",
]
