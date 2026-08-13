"""Physiology registry — the single source of truth for cell-type reference data.

The verification layers and the agent previously carried cell physiology as
scattered prose: the system prompt asserted "HepG2 doubling ~30–40 h" (wrong —
current catalogs say 48–60 h), the safety layer hard-coded a BSL-2 hint tuple,
and the calculators documented seeding densities ad hoc. That is how a number
drifts. This module is the one place cell-type reference data lives, and the
consumers derive from it:

- :func:`labwright.verify.safety.biosafety_level_for` — BSL level + hint.
- the ``cell_physiology`` tool — the agent looks physiology up instead of
  guessing it.
- :mod:`labwright.calc.o2` and :mod:`labwright.calc.barrier` — oxygen
  consumption and TEER/permeability reference ranges.
- :func:`physiology_anchor_text` — the physiological anchors rendered into the
  agent's system prompt, so the prompt cannot drift from the registry.

Every number is a *reference range*, never a fake-precision point estimate, and
each carries its source (DOI / catalog / convention label). Ranges are wide on
purpose: the registry tells a user what is plausible to check against, not what
their exact experiment will read.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CellProfile:
    """Reference physiology for one cell type / line."""

    key: str                                  # canonical key ("hepg2")
    aliases: tuple[str, ...] = ()
    organ: str = ""                           # organ/tissue the cell models
    human_primary: bool = False               # human-derived primary material
    animal_derived: bool = False              # non-human animal-derived cells
    bsl: int = 1                              # biosafety containment level
    bsl_hint: str | None = None               # reason, shown when bsl >= 2
    cell_diameter_um: float | None = None     # mean single-cell diameter
    doubling_time_h: tuple[float, float] | None = None  # (low, high) range
    doubling_note: str = ""                   # e.g. "primary cells do not divide"
    seeding_density_cells_cm2: tuple[float, float] | None = None  # (low, high)
    shear_range_pa: tuple[float, float] | None = None  # physiological shear, Pa
    o2_consumption_nmol_min_1e6: tuple[float, float] | None = None  # (low, high)
    teer_ohm_cm2: tuple[float, float] | None = None  # in-vitro model TEER range
    teer_physiological_ohm_cm2: tuple[float, float] | None = None  # in-vivo ref
    barrier: str | None = None                # e.g. "intestinal", "blood-brain"
    notes: str = ""
    sources: tuple[str, ...] = ()


#: Registry keyed by canonical key. Aliases are matched case-insensitively and
#: by substring, so "primary human hepatocytes", "HepG2 spheroids" and "Caco2"
#: all resolve. Numbers are ranges with sources; a bare "convention" label means
#: a textbook-level standard for which no single primary source is needed.
PROFILES: dict[str, CellProfile] = {
    "hepg2": CellProfile(
        key="hepg2",
        aliases=("hepg2", "hep-g2", "hepg2 cells"),
        organ="liver",
        cell_diameter_um=20.0,  # convention; matches the spheroid calculators
        doubling_time_h=(48, 60),
        doubling_note="catalogs report ~48 h (ENCODE4 SOP) to 50–60 h (DSMZ ACC-180); "
        "the ~30–40 h figure in older literature is not supported by current catalogs",
        seeding_density_cells_cm2=(5e4, 1e5),
        shear_range_pa=(0.05, 0.15),  # hepatic sinusoidal
        o2_consumption_nmol_min_1e6=(0.8, 4.7),
        notes="immortalized hepatoma line; not primary material",
        sources=(
            "DSMZ ACC-180 (https://www.dsmz.de/collection/catalogue/details/culture/ACC-180)",
            "ENCODE4 HepG2 growth SOP (~48 h doubling)",
            "Botte et al., Lab Chip 2024 — single-cell OCR, doi:10.1039/d4lc00204k",
        ),
    ),
    "heparg": CellProfile(
        key="heparg",
        aliases=("heparg", "heparg cells"),
        organ="liver",
        cell_diameter_um=20.0,  # convention
        doubling_time_h=(45, 55),
        doubling_note="~50 h population doubling (Gripon et al. 2002)",
        seeding_density_cells_cm2=(1e5, 3e5),
        sources=("Gripon et al., PNAS 2002 — HepaRG progenitor line",),
    ),
    "phh": CellProfile(
        key="phh",
        aliases=("phh", "primary human hepatocyte", "primary human hepatocytes",
                 "human primary hepatocyte", "human primary hepatocytes"),
        organ="liver",
        human_primary=True,
        bsl=2,
        bsl_hint="human-derived primary material warrants BSL-2 containment in most institutions",
        cell_diameter_um=20.0,  # convention
        doubling_time_h=None,
        doubling_note="primary human hepatocytes do not divide in vitro",
        seeding_density_cells_cm2=(1e5, 2e5),  # collagen-sandwich standard
        o2_consumption_nmol_min_1e6=(0.4, 4.7),
        notes="gold-standard liver cell; supply is limited and primary",
        sources=("Botte et al., Lab Chip 2024 — OCR range",),
    ),
    "primary-mouse-hepatocytes": CellProfile(
        key="primary-mouse-hepatocytes",
        aliases=("primary mouse hepatocyte", "primary mouse hepatocytes"),
        organ="liver",
        animal_derived=True,
        bsl=2,
        bsl_hint="primary hepatocyte material — confirm containment per institutional policy",
        cell_diameter_um=20.0,  # convention
        doubling_time_h=None,
        doubling_note="primary hepatocytes do not divide in vitro",
        seeding_density_cells_cm2=(1e5, 2e5),  # collagen-sandwich standard
        notes="animal-derived primary material",
        sources=("convention (primary hepatocyte sandwich culture)",),
    ),
    "hepatocyte-spheroid": CellProfile(
        key="hepatocyte-spheroid",
        aliases=("hepatocyte spheroid", "hepatocyte spheroids", "liver spheroid",
                 "liver spheroids", "primary hepatocyte spheroid"),
        organ="liver",
        human_primary=True,
        bsl=2,
        bsl_hint="spheroids from human primary material — handle under BSL-2 containment",
        cell_diameter_um=20.0,  # convention
        seeding_density_cells_cm2=None,
        o2_consumption_nmol_min_1e6=(4.7, 4.7),
        notes="~1000 primary hepatocytes ≈ a 200 µm spheroid (20 µm cells, dense "
        "packing); spheroids above ~400 µm develop necrotic cores (O2 diffuses "
        "~200 µm from the surface)",
        sources=(
            "Drug Metab Dispos 2024, doi:10.1124/dmd.124.001653 (necrotic cores)",
            "Botte et al., Lab Chip 2024 (spheroid OCR)",
        ),
    ),
    "hela": CellProfile(
        key="hela",
        aliases=("hela", "hela cells"),
        organ="cervix",
        bsl=2,
        bsl_hint="HeLa is classed BSL-2 at ATCC",
        cell_diameter_um=15.0,  # convention
        doubling_time_h=(24, 48),  # ATCC ~24 h; slower in practice
        sources=("ATCC HeLa BSL-2 classification",),
    ),
    "caco-2": CellProfile(
        key="caco-2",
        aliases=("caco-2", "caco2", "caco-2 cells"),
        organ="intestine",
        barrier="intestinal",
        cell_diameter_um=15.0,  # convention
        doubling_time_h=(20, 30),
        seeding_density_cells_cm2=(5e4, 1e5),
        teer_ohm_cm2=(250, 1000),
        teer_physiological_ohm_cm2=(250, 1200),
        notes="TEER ≥ ~300 Ω·cm² is the common QC gate before a transport study; "
        "mature 21-day monolayers run ~400–1200 Ω·cm²",
        sources=(
            "Caco-2 TEER review (ScienceDirect topic)",
            "Exp Mol Med 2025, doi:10.1038/s12276-025-01635-6 (Table 1: ~696 ± 126 Ω·cm²)",
        ),
    ),
    "hcmec-d3": CellProfile(
        key="hcmec-d3",
        aliases=("hcmec/d3", "hcmec d3", "hcmecd3", "hcmec-d3 cells"),
        organ="brain",
        barrier="blood-brain",
        human_primary=True,
        cell_diameter_um=15.0,  # convention
        teer_ohm_cm2=(100, 240),
        teer_physiological_ohm_cm2=(1800, 2000),
        notes="the hCMEC/D3 line forms only a moderately restrictive barrier "
        "(claudin-5 downregulated); even with co-culture/chemical enhancement it "
        "stays ~10× below physiological BBB resistance — the chip cannot reach "
        "in-vivo BBB TEER with this line",
        sources=(
            "Gericke et al., Fluids Barriers CNS 2020, doi:10.1186/s12987-020-00212-5",
        ),
    ),
    "huvec": CellProfile(
        key="huvec",
        aliases=("huvec", "huvec cells", "endothelial", "vascular endothelium"),
        organ="vasculature",
        cell_diameter_um=15.0,  # convention
        doubling_time_h=(24, 72),  # varies with passage
        seeding_density_cells_cm2=(1e4, 2e4),
        shear_range_pa=(0.1, 1.0),  # microvascular endothelium
        notes="HUVEC barrier/TEER is low (~20–100 Ω·cm²) and not the readout; "
        "endothelial health is assayed by morphology and junctional markers",
        sources=("convention (microvascular shear)",),
    ),
    "a549": CellProfile(
        key="a549",
        aliases=("a549", "a549 cells"),
        organ="lung",
        cell_diameter_um=15.0,  # convention
        doubling_time_h=(20, 26),
        seeding_density_cells_cm2=(5e3, 1e4),
        shear_range_pa=(0.02, 0.05),  # alveolar-capillary
        notes="alveolar epithelium line; barrier function is modest",
        sources=("convention (alveolar shear)",),
    ),
}

#: Generic containment hint when the string says "primary" but resolves to no
#: registered profile — any primary material warrants the conservative BSL-2
#: reminder rather than a silent BSL-1 default.
_PRIMARY_FALLBACK_HINT = "primary material — handle under BSL-2 containment in most institutions"


def lookup_cell(cell_type: str | None) -> CellProfile | None:
    """Resolve a cell-type string to a registry profile.

    Matching is case-insensitive, first by canonical key and alias exactly, then
    by longest-substring so "HepG2 spheroids", "Caco2" and "PHH" all resolve.
    ``None`` when nothing matches.
    """
    if not cell_type:
        return None
    q = " ".join(str(cell_type).strip().lower().split())
    for prof in PROFILES.values():
        if q == prof.key or q in prof.aliases:
            return prof
    best: tuple[int, CellProfile] | None = None
    for prof in PROFILES.values():
        for term in (prof.key, *prof.aliases):
            if term and term in q:
                if best is None or len(term) > best[0]:
                    best = (len(term), prof)
                break
    return best[1] if best else None


def biosafety_for(cell_type: str | None) -> tuple[int, str | None]:
    """(BSL level, hint) for a cell type, using the registry.

    Any unregistered string containing "primary" conservatively returns BSL-2
    (primary material is the highest-risk category). Everything else defaults to
    BSL-1 — the assignment of the common laboratory lines.
    """
    prof = lookup_cell(cell_type)
    if prof is not None:
        if prof.bsl >= 2:
            return prof.bsl, prof.bsl_hint
        return prof.bsl, None
    name = (cell_type or "").strip().lower()
    if "primary" in name:
        return 2, _PRIMARY_FALLBACK_HINT
    return 1, None


def physiology_anchor_text() -> str:
    """Render the registry's key anchors for the agent system prompt.

    The prompt imports this text instead of hard-coding physiology, so a
    corrected registry value (e.g. HepG2 doubling 48–60 h, not 30–40 h) reaches
    the agent automatically and the prose cannot drift from the data.
    """
    lines: list[str] = []
    for prof in PROFILES.values():
        desc = prof.organ or prof.barrier or "cell"
        bits: list[str] = []
        if prof.doubling_time_h is not None:
            bits.append(f"doubling {prof.doubling_time_h[0]:g}–{prof.doubling_time_h[1]:g} h")
        elif prof.doubling_note:
            bits.append(prof.doubling_note)
        if prof.seeding_density_cells_cm2 is not None:
            lo, hi = prof.seeding_density_cells_cm2
            bits.append(f"seed {lo:g}–{hi:g} cells/cm²")
        if prof.shear_range_pa is not None:
            lo, hi = prof.shear_range_pa
            bits.append(f"shear {lo:g}–{hi:g} Pa")
        if prof.teer_ohm_cm2 is not None:
            lo, hi = prof.teer_ohm_cm2
            bits.append(f"model TEER {lo:g}–{hi:g} Ω·cm²")
        if prof.human_primary:
            bits.append("human primary — BSL-2")
        lines.append(f"- {prof.key} ({desc}): {', '.join(bits) if bits else 'reference data'}.")
    return "\n".join(lines)


__all__ = [
    "CellProfile",
    "PROFILES",
    "lookup_cell",
    "biosafety_for",
    "physiology_anchor_text",
]
