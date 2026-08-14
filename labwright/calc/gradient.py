"""Steady concentration-gradient calculators for chemotaxis experiments.

Neutrophil, dendritic-cell, cancer and axon-guidance assays all need a
*stable, near-linear* concentration gradient of a chemoattractant across the
cell culture region. Two families of on-chip generators produce those
gradients:

1. **Diffusion-based source-sink designs** — a chemoattractant is perfused
   through a *source* channel and buffer through a *sink* channel, with a
   stagnant agarose (or porous membrane / PDMS) bridge between them. Solute
   diffuses from source to sink and, at steady state, sets up a near-linear
   Fick profile across the gap; cells are cultured on or in the bridge. The
   classic geometry is a ~200 µm agarose layer with source/sink channels
   ~1 mm apart edge-to-edge.
2. **Laminar-flow gradient generators** — two (or more) streams perfused side
   by side in a single channel diffuse across the stream interface and set up a
   gradient along the channel width. These stabilise fast (~3 min) and hold for
   ≥60 min while flow runs, but they consume solute continuously and need a
   pump.

This module supplies the steady-state arithmetic for the first family and the
design checks (relaxation time, flux, stability window) a copilot needs to
answer "will the gradient be established and held over my experiment?", plus
the inverse problem — what channel spacing hits a target steepness.

Units are plain floats named by the argument docstrings: concentrations in µM,
distances in µm, diffusivity in m²/s, flux in mol/m²/s. 1 µM = 1e-3 mol/m³.

References
----------
- Stable chemokine gradients for dendritic-cell chemotaxis, Frontiers in Cell
  and Developmental Biology 2022,
  https://www.frontiersin.org/journals/cell-and-developmental-biology/articles/10.3389/fcell.2022.943041/full
  (flow-based generator: gradient stabilization ~3 min, stable ≥60 min).
- Generating 2-dimensional concentration gradients using a simple microfluidic
  design (EuropePMC), https://pmc.ncbi.nlm.nih.gov/articles/PMC5552394/
  (diffusion-based source-sink: ~200 µm agarose layer, edge-to-edge channel
  spacing ~1 mm).
- Small-molecule / FITC diffusivity in water/agarose ≈ 1e-9 to 1e-10 m²/s —
  order-of-magnitude estimate from standard tables; the mid-value 5e-10 m²/s
  is used as the module default.
"""

from __future__ import annotations

import math

#: Small-molecule / FITC diffusivity in water/agarose, m²/s. Order-of-magnitude
#: estimate (1e-9 to 1e-10) from standard tables; 5e-10 is the mid-value used
#: as the default for a diffusive agarose-bridge generator.
SMALL_MOLECULE_DIFFUSIVITY_M2S = 5e-10


# ---------------------------------------------------------------------------
# Steady gradient profile
# ---------------------------------------------------------------------------


def linear_gradient_steepness_um_per_mm(
    source_conc_um: float,
    sink_conc_um: float,
    distance_um: float,
) -> float:
    """Linear gradient steepness across the source-sink gap, µM per mm.

    .. math:: \\frac{dC}{dx} = \\frac{C_{src} - C_{sink}}{L}\\, \\times 1000

    The steady-state concentration drop per unit distance — the number a
    chemotaxis protocol quotes ("a 90 µM/mm CXCL12 gradient").

    Parameters
    ----------
    source_conc_um : float
        Chemoattractant concentration in the source channel, µM.
    sink_conc_um : float
        Buffer concentration in the sink channel, µM.
    distance_um : float
        Source-to-sink separation (the diffusive gap), µm.

    Returns
    -------
    float
        Steepness in µM per mm. With a linear profile the midpoint of the gap
        reads ``(C_src + C_sink)/2``.
    """
    if distance_um <= 0:
        raise ValueError(f"distance_um must be > 0, got {distance_um!r}")
    if not math.isfinite(float(source_conc_um)) or source_conc_um < 0:
        raise ValueError(f"source_conc_um must be finite and >= 0, got {source_conc_um!r}")
    if not math.isfinite(float(sink_conc_um)) or sink_conc_um < 0:
        raise ValueError(f"sink_conc_um must be finite and >= 0, got {sink_conc_um!r}")
    return (source_conc_um - sink_conc_um) / distance_um * 1000.0


def steady_state_profile_conc_um(
    source_conc_um: float,
    sink_conc_um: float,
    distance_um: float,
    x_um: float,
) -> float:
    """Steady-state Fick profile across a source-sink gap, µM.

    With constant boundary concentrations the diffusive profile between source
    and sink is linear:

    .. math:: C(x) = C_{src} - \\frac{C_{src} - C_{sink}}{L}\\, x

    Parameters
    ----------
    source_conc_um : float
        Chemoattractant concentration at the source channel, µM.
    sink_conc_um : float
        Buffer concentration at the sink channel, µM.
    distance_um : float
        Source-to-sink separation, µm.
    x_um : float
        Position along the gap measured from the source, µm
        (0 ≤ x ≤ distance).

    Returns
    -------
    float
        Concentration at ``x_um``, µM. ``C(0) = C_src``, ``C(L) = C_sink``.
    """
    if distance_um <= 0:
        raise ValueError(f"distance_um must be > 0, got {distance_um!r}")
    if not math.isfinite(float(x_um)) or x_um < 0 or x_um > distance_um:
        raise ValueError(f"x_um must lie in [0, distance_um], got {x_um!r}")
    if not math.isfinite(float(source_conc_um)) or source_conc_um < 0:
        raise ValueError(f"source_conc_um must be finite and >= 0, got {source_conc_um!r}")
    if not math.isfinite(float(sink_conc_um)) or sink_conc_um < 0:
        raise ValueError(f"sink_conc_um must be finite and >= 0, got {sink_conc_um!r}")
    return source_conc_um - (source_conc_um - sink_conc_um) * x_um / distance_um


# ---------------------------------------------------------------------------
# Dynamics and flux
# ---------------------------------------------------------------------------


def diffusive_relaxation_time_s(
    distance_um: float,
    diffusivity_m2s: float = SMALL_MOLECULE_DIFFUSIVITY_M2S,
) -> float:
    """Time to establish the diffusive gradient, s.

    .. math:: \\tau \\approx \\frac{L^2}{D}

    The diffusion time across the gap: a source-sink gradient takes roughly
    ``τ`` to form after the source is switched on and re-forms in ``τ`` after a
    perturbation. For a 1 mm agarose bridge at small-molecule diffusivity this
    is ~30 min.

    Parameters
    ----------
    distance_um : float
        Source-to-sink separation, µm.
    diffusivity_m2s : float, default 5e-10
        Solute diffusivity in the bridge, m²/s (small-molecule/FITC in
        water/agarose estimate; 1e-9 to 1e-10 order of magnitude).

    Returns
    -------
    float
        Relaxation time in seconds.
    """
    if distance_um <= 0:
        raise ValueError(f"distance_um must be > 0, got {distance_um!r}")
    if not math.isfinite(float(diffusivity_m2s)) or diffusivity_m2s <= 0:
        raise ValueError(f"diffusivity_m2s must be finite and > 0, got {diffusivity_m2s!r}")
    length_m = distance_um * 1e-6
    return length_m**2 / diffusivity_m2s


def diffusive_flux_mol_m2s(
    source_conc_um: float,
    sink_conc_um: float,
    distance_um: float,
    diffusivity_m2s: float = SMALL_MOLECULE_DIFFUSIVITY_M2S,
) -> float:
    """Steady-state diffusive flux (Fick's first law), mol/m²/s.

    .. math:: J = D\\, \\frac{C_{src} - C_{sink}}{L}

    Concentrations convert as 1 µM = 1e-3 mol/m³ and ``L`` from µm to m. The
    flux is what the source channel must supply (per membrane area) to hold the
    gradient — a design input for flow rate and solute budget.

    Parameters
    ----------
    source_conc_um : float
        Chemoattractant concentration at the source channel, µM.
    sink_conc_um : float
        Buffer concentration at the sink channel, µM.
    distance_um : float
        Source-to-sink separation, µm.
    diffusivity_m2s : float, default 5e-10
        Solute diffusivity in the bridge, m²/s (small-molecule estimate).

    Returns
    -------
    float
        Flux in mol/m²/s.
    """
    if distance_um <= 0:
        raise ValueError(f"distance_um must be > 0, got {distance_um!r}")
    if not math.isfinite(float(source_conc_um)) or source_conc_um < 0:
        raise ValueError(f"source_conc_um must be finite and >= 0, got {source_conc_um!r}")
    if not math.isfinite(float(sink_conc_um)) or sink_conc_um < 0:
        raise ValueError(f"sink_conc_um must be finite and >= 0, got {sink_conc_um!r}")
    if not math.isfinite(float(diffusivity_m2s)) or diffusivity_m2s <= 0:
        raise ValueError(f"diffusivity_m2s must be finite and > 0, got {diffusivity_m2s!r}")
    c_src_mol_m3 = source_conc_um * 1e-3  # 1 µM = 1e-3 mol/m³
    c_sink_mol_m3 = sink_conc_um * 1e-3
    length_m = distance_um * 1e-6
    return diffusivity_m2s * (c_src_mol_m3 - c_sink_mol_m3) / length_m


def gradient_stability_check(
    relaxation_time_s: float,
    experiment_hours: float,
) -> dict[str, float]:
    """Whether an experiment holds its gradient long enough to trust.

    A diffusive gradient is "stable" when the experiment both establishes it
    (≈ τ) and holds it well past that (≫ τ):

    .. math:: t_{exp} = \\text{hours} \\times 3600 \\ge 10\\,\\tau

    The 10τ rule guarantees the profile has reached steady state and then been
    sustained for a large multiple of the relaxation time, so the biological
    readout samples a stable gradient rather than its establishment transient.

    Parameters
    ----------
    relaxation_time_s : float
        Gradient relaxation time, s (see :func:`diffusive_relaxation_time_s`).
    experiment_hours : float
        Planned experiment duration, hours.

    Returns
    -------
    dict
        ``{"tau_s", "hours", "stable"}`` where ``stable`` is True when the
        experiment runs ≥ 10τ.
    """
    if not math.isfinite(float(relaxation_time_s)) or relaxation_time_s <= 0:
        raise ValueError(f"relaxation_time_s must be finite and > 0, got {relaxation_time_s!r}")
    if not math.isfinite(float(experiment_hours)) or experiment_hours <= 0:
        raise ValueError(f"experiment_hours must be finite and > 0, got {experiment_hours!r}")
    experiment_s = experiment_hours * 3600.0
    return {
        "tau_s": round(relaxation_time_s, 6),
        "hours": round(experiment_hours, 6),
        "stable": bool(experiment_s >= 10.0 * relaxation_time_s),
    }


# ---------------------------------------------------------------------------
# Inverse design
# ---------------------------------------------------------------------------


def source_sink_channel_spacing_um_from_gradient(
    target_steepness_um_per_mm: float,
    source_conc_um: float,
    sink_conc_um: float,
) -> float:
    """Source-sink channel spacing for a target steepness, µm.

    .. math:: L = \\frac{C_{src} - C_{sink}}{dC/dx}\\, \\times 1000

    Inverse of :func:`linear_gradient_steepness_um_per_mm` — "if I want a
    90 µM/mm CXCL12 gradient from a 100 µM source against buffer, how far apart
    do the channels sit?" The classic ~1 mm spacing follows from a ~0.1 mM
    source-to-buffer drop at ~0.1 mM/mm.

    Parameters
    ----------
    target_steepness_um_per_mm : float
        Desired gradient steepness, µM per mm.
    source_conc_um : float
        Chemoattractant concentration in the source channel, µM.
    sink_conc_um : float
        Buffer concentration in the sink channel, µM.

    Returns
    -------
    float
        Source-to-sink edge spacing in µm.
    """
    if target_steepness_um_per_mm <= 0:
        raise ValueError(f"target_steepness_um_per_mm must be > 0, got {target_steepness_um_per_mm!r}")
    if not math.isfinite(float(source_conc_um)) or source_conc_um < 0:
        raise ValueError(f"source_conc_um must be finite and >= 0, got {source_conc_um!r}")
    if not math.isfinite(float(sink_conc_um)) or sink_conc_um < 0:
        raise ValueError(f"sink_conc_um must be finite and >= 0, got {sink_conc_um!r}")
    if source_conc_um <= sink_conc_um:
        raise ValueError(
            f"source_conc_um must exceed sink_conc_um for a source-sink "
            f"gradient, got source={source_conc_um!r}, sink={sink_conc_um!r}"
        )
    return (source_conc_um - sink_conc_um) * 1000.0 / target_steepness_um_per_mm


__all__ = [
    "SMALL_MOLECULE_DIFFUSIVITY_M2S",
    "linear_gradient_steepness_um_per_mm",
    "steady_state_profile_conc_um",
    "diffusive_relaxation_time_s",
    "diffusive_flux_mol_m2s",
    "gradient_stability_check",
    "source_sink_channel_spacing_um_from_gradient",
]
