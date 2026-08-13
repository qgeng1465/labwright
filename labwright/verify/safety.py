"""Safety & compliance layer — configurable, never silently passed.

The arithmetic, unit and range checks prove numbers are *right*; the safety
layer asks whether the *experiment itself* is safe to run. Three
responsibilities:

1. **Solvent & drug-dose guardrails** — DMSO carry-over (configurable) and
   guidance caps on a small set of compounds whose in-vitro toxicity threshold
   is well established. Out-of-bound is a warning (guidance cap) or an error
   (hard rejection) with the reason attached — never silent.
2. **Biosafety hints** — a conservative BSL assignment for common cell types,
   warning when the material warrants containment (human-derived primary
   material, BSL-2 lines).
3. **Institutional safety boundary** — every threshold lives in
   :class:`SafetyConfig`; a lab tightens or relaxes the boundary there instead
   of forking the verifier.

.. note::
    None of these thresholds is medical or regulatory advice. They are
    deliberately conservative *guidance defaults* with sources noted below,
    which a laboratory must reconcile against its own SOPs and ethics board.
    ``SafetyConfig`` is the override point.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from labwright.schema.design import DesignPlan

# ---------------------------------------------------------------------------
# Configuration — the institution's safety boundary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SafetyConfig:
    """A laboratory's configurable safety boundary."""

    #: Max DMSO volume fraction in the final medium (v/v). 0.005 = 0.5 % v/v,
    #: the conventional upper end of the safe window for most adherent cells.
    max_dmso_vv: float = 0.005
    #: Demand a matched vehicle control whenever a compound is dosed.
    require_vehicle_control: bool = True
    #: Guidance caps per compound (mM), merged over :data:`CHEMICAL_LIMITS`.
    #: A lab can add its own entries or tighten/relax the built-in ones.
    chemical_limits_mM: dict[str, dict] = field(default_factory=dict)
    #: Emit biosafety-containment hints (BSL assignment) for cell types.
    biosafety_hints: bool = True
    #: Emit animal-ethics reminders for animal-derived cell types.
    animal_ethics_reminders: bool = True
    #: Free-text institutional note appended to safety findings (e.g. "C-301").
    institution: str = ""


#: Module-level safety boundary. Importers read/write it through the helpers.
_DEFAULT = SafetyConfig()
_ACTIVE = _DEFAULT


def get_safety_config() -> SafetyConfig:
    """The safety boundary currently in force."""
    return _ACTIVE


def set_safety_config(config: SafetyConfig) -> None:
    """Replace the safety boundary in force (e.g. from a lab's JSON config)."""
    global _ACTIVE
    _ACTIVE = config


def reset_safety_config() -> None:
    """Restore the built-in guidance defaults."""
    global _ACTIVE
    _ACTIVE = _DEFAULT


def load_safety_config(path: str) -> SafetyConfig:
    """Load a safety boundary from JSON. Unknown keys are ignored."""
    import json

    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    known = {k: v for k, v in data.items() if k in SafetyConfig.__dataclass_fields__}
    cfg = SafetyConfig(**known)
    set_safety_config(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Compound guidance caps (mM working concentration)
# ---------------------------------------------------------------------------

#: Guidance caps with a defensible in-vitro basis. ``guidance_mM`` triggers a
#: warning (above the threshold that typically starts to harm cells);
#: ``reject_mM`` is a hard error (far beyond any published in-vitro dose). A lab
#: that runs these compounds routinely should set its own caps in SafetyConfig.
CHEMICAL_LIMITS: dict[str, dict] = {
    "acetaminophen": {
        "aliases": ("acetaminophen", "apap", "paracetamol", "n-acetyl-p-aminophenol"),
        "guidance_mM": 10.0,
        "reject_mM": 100.0,
        "basis": "in-vitro hepatotoxicity onset ~5–10 mM (HepG2 / primary human hepatocytes)",
    },
    "doxorubicin": {
        "aliases": ("doxorubicin", "adriamycin"),
        "guidance_mM": 0.01,  # 10 µM
        "reject_mM": 0.5,
        "basis": "in-vitro cytotoxic range ~0.01–1 µM for most carcinoma lines",
    },
}


def _caps_for(compound: str) -> dict | None:
    name = (compound or "").strip().lower()
    for key, entry in CHEMICAL_LIMITS.items():
        if name == key or name in {a.lower() for a in entry["aliases"]}:
            return entry
    return None


# ---------------------------------------------------------------------------
# Biosafety hint
# ---------------------------------------------------------------------------

_BSL_2_HINTS = (
    ("hela", "HeLa is classed BSL-2 at ATCC"),
    ("primary", "human-derived primary material warrants BSL-2 containment in most institutions"),
    ("phh", "primary human hepatocytes — human-derived, handle under BSL-2 containment"),
)


def biosafety_level_for(cell_type: str) -> tuple[int, str | None]:
    """Conservative BSL assignment for a cell type: ``(level, hint)``.

    Defaults to BSL-1 (the assignment of the common laboratory lines HepG2,
    Caco-2, HUVEC, A549). Only clearly documented exceptions are raised to
    BSL-2. The hint is emitted as a warning so the human confirms handling.
    """
    name = (cell_type or "").strip().lower()
    for token, hint in _BSL_2_HINTS:
        if token in name:
            return 2, hint
    return 1, None


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------


def check_safety(plan: DesignPlan, issues: list, config: SafetyConfig | None = None) -> None:
    """Append a safety issue for every guardrail the design crosses.

    - DMSO carry-over above ``config.max_dmso_vv`` → warning (solvent toxicity).
    - A dosed compound over its guidance cap → warning; over its reject cap →
      error ("reject with reason").
    - Dosing without a matched vehicle control (when required) → warning.
    - BSL-2+ cell material → warning (containment).
    - Animal-derived cell material → warning (ethics reminder).

    Configuration comes from ``config``, else the module-level boundary.
    """
    from labwright.verify.checker import Issue  # lazy: avoid a module cycle

    cfg = config if config is not None else _ACTIVE
    issues_to_add: list[Issue] = []

    # --- solvent / dose guardrails -----------------------------------------
    if plan.dosing is not None:
        if plan.dosing.dmso_fraction_vv > cfg.max_dmso_vv:
            issues_to_add.append(Issue(
                level="warning",
                field="dosing.dmso_fraction_vv",
                message=(
                    f"DMSO {plan.dosing.dmso_fraction_vv * 100:.2f}% v/v exceeds the "
                    f"safety boundary {cfg.max_dmso_vv * 100:.2f}% v/v "
                    "(solvent toxicity) — use a more concentrated stock or a lower dose"
                ),
            ))
        caps = _caps_for(plan.dosing.compound)
        if caps:
            merged = dict(caps)
            merged.update(cfg.chemical_limits_mM.get(plan.dosing.compound.strip().lower(), {}))
            working = plan.dosing.working_mM
            if working > merged.get("reject_mM", float("inf")):
                issues_to_add.append(Issue(
                    level="error",
                    field="dosing.working_mM",
                    message=(
                        f"{plan.dosing.compound} working dose {working:g} mM exceeds the "
                        f"hard safety cap {merged['reject_mM']:g} mM and is rejected "
                        f"({merged.get('basis', 'no basis stated')})"
                    ),
                ))
            elif working > merged.get("guidance_mM", float("inf")):
                issues_to_add.append(Issue(
                    level="warning",
                    field="dosing.working_mM",
                    message=(
                        f"{plan.dosing.compound} working dose {working:g} mM is above the "
                        f"guidance cap {merged['guidance_mM']:g} mM "
                        f"({merged.get('basis', 'no basis stated')}) — confirm the dose "
                        f"with your SOP"
                    ),
                ))
        if cfg.require_vehicle_control and not plan.dosing.vehicle_control:
            issues_to_add.append(Issue(
                level="warning",
                field="dosing.vehicle_control",
                message="compound dosed without a matched vehicle control — solubility/solvent "
                "effects will be uninterpretable",
            ))

    # --- biosafety / ethics -------------------------------------------------
    cell_type = None
    if plan.cells is not None:
        cell_type = plan.cells.cell_type
        field = "cells.cell_type"
    elif plan.culture is not None:
        cell_type = plan.culture.cell_type
        field = "culture.cell_type"
    if cell_type:
        level, hint = biosafety_level_for(cell_type)
        if cfg.biosafety_hints and level >= 2 and hint:
            issues_to_add.append(Issue(
                level="warning",
                field=field,
                message=f"{cell_type}: {hint} — confirm containment before ordering",
            ))
        if cfg.animal_ethics_reminders and any(
            token in cell_type.lower() for token in ("mouse", "rat", "rabbit", "porcine", "canine", "primate", "bovine")
        ):
            issues_to_add.append(Issue(
                level="warning",
                field=field,
                message=f"animal-derived cells ({cell_type}) — confirm the institution's animal-ethics "
                "approval covers this work",
            ))

    # Institutional note (config) rides on every safety finding.
    for issue in issues_to_add:
        if cfg.institution:
            issue = replace(issue, message=f"{issue.message} [{cfg.institution}]")
        issues.append(issue)


__all__ = [
    "SafetyConfig",
    "CHEMICAL_LIMITS",
    "get_safety_config",
    "set_safety_config",
    "reset_safety_config",
    "load_safety_config",
    "biosafety_level_for",
    "check_safety",
]
