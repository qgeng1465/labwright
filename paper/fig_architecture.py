"""Render the Labwright architecture figure: a paneled (a)-(e) system anatomy.

This is a NEW figure, deliberately structured like the BPL-COGEN Fig. 1 legend
(panelled: layered architecture / bounded workflow / compiler pipeline) rather
than the single-column main-flow + detail-panel layout of ``fig_pipeline.py``.
It is meant to be the figure a reader stares at to understand *what Labwright
is made of* — every layer, every calculator, every internal agent.

Panels
------
(a)  Layered architecture — 8 layers: input boundary -> agent brain -> tool
     registry -> calculator core -> domain blocks -> verifier -> knowledge
     base -> outputs. The boundary rule (LLM proposes raw inputs; calculators
     compute; the verifier re-proves) is the top line.
(b)  Bounded agentic workflow — the numbered 5-step ReAct flow, its retry lane
     (fix raw inputs only), and the benchmark ablation row.
(c)  Verifier pipeline — the 5 serial layers, each with its verdict semantics.
(d)  Calculator toolbox — all 46 tools in their 10 classes (enumerated from
     ``labwright.tools.REGISTRY``), plus the calc/ modules and formulas.
(e)  Internal components, benchmark systems and the honest boundary.

Every number and component shown is real and committed (eval/README.md + the
committed results/). Nothing speculative is drawn. Editing hooks: the 46-tool
table is one dict at the top and the benchmark numbers are one ``BENCH`` dict,
so later finetune runs only need one-line updates.

Usage::

    python paper/fig_architecture.py   # writes paper/fig_architecture.pdf/.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _font import setup_font  # noqa: E402

setup_font()

# ---------------------------------------------------------------------------
# Palette — shared with fig_pipeline / fig_abstract so the system identity is
# one colour everywhere.
# ---------------------------------------------------------------------------
INK = "#262522"
MUT = "#8a8782"
GRID = "#d9d7d3"
BLUE = "#2E5598"          # Labwright main path
BLUE_EDGE = "#1f3f70"
BLUE_LIGHT = "#c3d2ec"    # goal entry
GRAY = "#a8a39d"
GRAY_LIGHT = "#e0dcd5"
GRAY_DARK = "#74706a"
RED = "#b3261e"           # status: hard reject / error
PALE = "#f6f4f0"          # detail-panel fill

# ---------------------------------------------------------------------------
# EDITING HOOK 1 — the whole tool registry, one dict. Panels (a) and (d) derive
# from it. (Names are exactly the register_tool() names in labwright/tools.py.)
# ---------------------------------------------------------------------------
TOOLS = {
    "microfluidics": ["wall_shear_stress", "flow_rate_for_shear_stress",
                      "reynolds_number", "pressure_drop", "residence_time",
                      "o2_delivery_rate"],
    "cell": ["seeding_cell_count", "cell_count_after_time", "time_to_confluence",
             "cells_per_well", "medium_volume_per_well", "hemocytometer_count",
             "trypan_blue_viability", "confluence_to_cell_count",
             "cell_count_to_confluence", "passage_split_ratio", "moi_virus_volume"],
    "dosing": ["molarity_from_mass", "dilution_volume", "dmso_fraction"],
    "stats": ["sample_size_per_group", "technical_replicates"],
    "published": ["verify_published_protocol"],
    "3d_culture": ["spheroid_volume", "spheroid_diameter_from_cells",
                   "cells_per_spheroid_for_diameter", "medium_volume_per_spheroid",
                   "spheroids_from_suspension"],
    "physiology": ["cell_physiology"],
    "o2": ["o2_penetration_depth", "spheroid_necrotic_fraction",
           "o2_supply_vs_demand", "o2_peclet", "o2_damkohler", "o2_po2_conversion"],
    "barrier": ["teer_from_resistance", "transendothelial_resistance",
                "papp_from_flux", "flux_from_papp", "clearance_from_papp",
                "effective_permeability"],
    "pk": ["extraction_ratio", "clearance_uLmin", "half_life_h",
           "accumulation_ratio", "mass_cleared_ug_h"],
}

# ---------------------------------------------------------------------------
# EDITING HOOK 2 — benchmark numbers, one dict. All committed values.
# ---------------------------------------------------------------------------
BENCH = {
    "reading_usable": "88% flash / 100% pro",          # 24-reading gold set
    "reading_ci": "5-seed Wilson 0.925 [0.864, 0.960] / 0.958 [0.906, 0.982]",
    "reading_halluc": "0.125 / 0.000",
    "blind_usable": "40% / 47%  (cold-only 38%)",      # 15-blind gold set
    "blind_halluc": "0.000 (attack-tested) · self-consistency 100%",
    "extractor": "JSON parse 1.0 · extract->verify 0.9976 vs untuned 0.40",
}

# Canvas: tall poster, 5 stacked panels. Data units == inches (figsize width
# 12.5 == xlim 12.5), so char-width budgeting below is in inches.
X0, X1, Y0, Y1 = 0.0, 12.5, 0.0, 13.2
N_TOOLS = sum(len(v) for v in TOOLS.values())
N_CLASSES = len(TOOLS)


def _charcap(width_in: float, fs: float) -> int:
    """Max chars that fit ``width_in`` inches of text at size ``fs`` (pt).

    TNR proportional text averages ~0.5*fs pt per char; ``width_in`` is in data
    units == inches. A 0.95 safety factor keeps text off the box edges.
    """
    return int((width_in * 72 / (0.5 * fs)) * 0.95)


def _par_wrap(text: str, width: int) -> str:
    """Wrap every paragraph (split by ``\\n``) to ``width`` chars each."""
    out: list[str] = []
    for para in text.split("\n"):
        words = para.split()
        if not words:
            out.append("")
            continue
        lines: list[str] = []
        cur = ""
        for w in words:
            cand = (cur + " " + w).strip()
            if len(cand) <= width:
                cur = cand
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
        out.append("\n".join(lines))
    return "\n".join(out)


def _box(ax, x, y, w, h, text, fc, ec, tc=INK, fs=7.5, weight="normal",
         line_h=1.2, zorder=3, pad=0.16, cap=None):
    ax.add_patch(FancyBboxPatch(
        (x + pad, y + pad), w - 2 * pad, h - 2 * pad,
        boxstyle=f"round,pad={pad},rounding_size=0.10",
        fc=fc, ec=ec, linewidth=1.1, zorder=zorder))
    if cap is None:
        cap = _charcap(w - 2 * pad, fs)
    ax.text(x + w / 2, y + h / 2, _par_wrap(text, cap), ha="center", va="center",
            fontsize=fs, color=tc, zorder=zorder + 1, fontweight=weight,
            linespacing=line_h)


def _panel(ax, x0, y0, x1, y1, title, fs=7.8, zorder=1):
    """A light detail panel with a small top-left title."""
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0.15,rounding_size=0.08",
                                fc=PALE, ec=GRID, linewidth=1.0, zorder=zorder))
    ax.text(x0 + 0.18, y1 - 0.12, title, fontsize=fs, color=BLUE_EDGE,
            ha="left", va="top", weight="bold", zorder=zorder + 1)


def _arrow(ax, x1, y1, x2, y2, color=MUT, style="-|>", lw=1.3, zorder=2, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=12, color=color, linewidth=lw,
                                 zorder=zorder, linestyle=ls))


def _label(ax, x, y, text, fs=6.8, color=MUT, ha="left", va="bottom",
           weight="normal", zorder=2, line_h=1.2, cap=None):
    if cap is not None:
        text = _par_wrap(text, cap)
    ax.text(x, y, text, fontsize=fs, color=color, ha=ha, va=va, zorder=zorder,
            fontweight=weight, linespacing=line_h)


def check_overlaps(fig, ax) -> list[str]:
    """Return a list of layout problems (text-text overlap / canvas spill)."""
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    problems: list[str] = []
    boxes = []
    for i, t in enumerate(ax.texts):
        bb = t.get_window_extent(renderer=renderer)
        x0, y0 = inv.transform((bb.x0, bb.y0))
        x1, y1 = inv.transform((bb.x1, bb.y1))
        boxes.append((i, t.get_text().replace("\n", "\\n")[:34], x0, y0, x1, y1))
        if x0 < X0 - 0.02 or y0 < Y0 - 0.02 or x1 > X1 + 0.02 or y1 > Y1 + 0.02:
            problems.append(
                f"  text[{i}] {boxes[-1][1]!r} spills canvas "
                f"({x0:.2f},{y0:.2f})-({x1:.2f},{y1:.2f})")
    for a in range(len(boxes)):
        for b in range(a + 1, len(boxes)):
            ax0, ay0, ax1, ay1 = boxes[a][2:]
            bx0, by0, bx1, by1 = boxes[b][2:]
            ox = max(0.0, min(ax1, bx1) - max(ax0, bx0))
            oy = max(0.0, min(ay1, by1) - max(ay0, by0))
            if ox > 0.01 and oy > 0.01:
                problems.append(
                    f"  text[{a}] {boxes[a][1]!r} OVERLAPS text[{b}] "
                    f"{boxes[b][1]!r} ({ox:.2f}x{oy:.2f})")
    return problems


def main() -> int:
    fig, ax = plt.subplots(figsize=(12.5, 13.2))
    ax.set_xlim(X0, X1)
    ax.set_ylim(Y0, Y1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # =====================================================================
    # (a)  LAYERED ARCHITECTURE
    # =====================================================================
    _panel(ax, 0.1, 9.95, 12.4, 13.0,
           "(a)  layered architecture — the LLM proposes raw inputs; "
           "deterministic calculators compute; the verifier re-proves")
    _arrow(ax, 0.72, 12.62, 0.72, 10.32, color=BLUE_EDGE, lw=1.6)

    layers = [
        ("L1  input boundary",
         "goal prose -> DesignInput (raw only, extra=\"forbid\"); "
         "_reject_derived_fields rejects any derived key — a typed number is not trusted"),
        ("L2  agent brain",
         "LLMClient (deepseek-v4-flash / pro, T=0.2, thinking off, any OpenAI-compatible "
         "endpoint) + DesignAgent ReAct loop (12 iters, 8 tool calls/turn)"),
        ("L3  tool registry",
         "46 calculators + submit_design gate, 10 classes; pydantic params -> JSON Schema; "
         "each tool carries a worked example + common unit trap"),
        ("L4  calculator core",
         "10 pure calc/ modules (microfluidics · cell · culture · spheroid · o2 · barrier · "
         "dosing · pk · stats · units) + pint; derived numbers are produced ONLY here"),
        ("L5  domain blocks",
         "blocks.py: 7 declarative Blocks (flow/cells/culture/spheroid/dosing/stats/pk) -> "
         "ALL_* tables of keys, sanity bands, canonical units; import-time validation"),
        ("L6  verifier",
         "5 layers re-derive every derived number from the agent's own inputs: arithmetic -> "
         "units -> sanity -> safety -> prose numbers"),
        ("L7  knowledge base",
         "physiology.py: 10 cell profiles (shear · TEER · O2 · BSL, with sources) anchored "
         "into the prompt; plate-format & spheroid-format tables"),
        ("L8  outputs",
         "DesignPlan JSON · Markdown SOP · provenance/ELN (formula + inputs + units + code "
         "version; json | csv) · Gradio UI · CLI"),
    ]
    ly_top, ly_bot, gap = 12.55, 10.25, 0.04
    band_h = (ly_top - ly_bot - (len(layers) - 1) * gap) / len(layers)
    for i, (name, detail) in enumerate(layers):
        yb = ly_top - band_h - i * (band_h + gap)
        _box(ax, 1.1, yb, 2.35, band_h, name, BLUE, BLUE_EDGE, tc="white",
             fs=6.3, weight="bold", pad=0.05)
        _label(ax, 3.65, yb + band_h / 2 - 0.055, detail, fs=6.15, color=INK,
               va="center", cap=_charcap(8.5, 6.15))
        _label(ax, 0.98, yb + band_h / 2 - 0.045, f"{i + 1}", fs=5.6, color=INK,
               ha="center", va="center", weight="bold")

    # =====================================================================
    # (b)  BOUNDED AGENTIC WORKFLOW
    # =====================================================================
    _panel(ax, 0.1, 7.40, 12.4, 9.95,
           "(b)  bounded agentic workflow — one goal in, a design whose every number "
           "was computed by calc/ and re-proved by verify/ comes out")
    flow = [
        ("1", "goal", "wet-lab goal in natural language + assumptions"),
        ("2", "ReAct tool loop", "LLM proposes raw inputs; calls calculators to reason; "
         "prose answers are refused"),
        ("3", "submit_design", "hard gate: derived fields rejected; DesignInput extra=forbid"),
        ("4", "verifier re-proves", "ok -> accept · review_required -> fix raw inputs only"),
        ("5", "SOP + design JSON", "verified protocol + provenance / ELN export"),
    ]
    bx0, bw, bgap, bh = 0.3, 2.28, 0.13, 0.82
    by = 8.72
    for i, (num, head, desc) in enumerate(flow):
        x = bx0 + i * (bw + bgap)
        _box(ax, x, by, bw, bh, f"{num}. {head}\n{desc}", BLUE, BLUE_EDGE,
             tc="white", fs=6.6, pad=0.09)
    for i in range(len(flow) - 1):
        _arrow(ax, bx0 + (i + 1) * bw + i * bgap + 0.015, by + bh / 2,
               bx0 + (i + 1) * (bw + bgap) - 0.03, by + bh / 2, color=BLUE_EDGE)

    # retry lane — review_required / validation_error loops back to the agent
    lane_y = 8.18
    lane_x1 = bx0 + 3 * (bw + bgap) + bw / 2   # under box 4 (verifier re-proves)
    lane_x2 = bx0 + bw / 2                     # under box 2 (ReAct tool loop)
    _arrow(ax, lane_x1, by - 0.02, lane_x1, lane_y + 0.03, color=RED, lw=1.2,
           ls=(0, (4, 2)))
    _arrow(ax, lane_x1, lane_y, lane_x2, lane_y, color=RED, lw=1.2, ls=(0, (4, 2)))
    _arrow(ax, lane_x2, lane_y, lane_x2, by - 0.04, color=RED, lw=1.2, ls=(0, (4, 2)))
    _label(ax, bx0 + 1.75 * (bw + bgap), lane_y + 0.16,
           "review_required / validation_error -> the agent fixes ONLY the raw inputs "
           "it proposed (never hand-writes a derived number)",
           fs=6.4, color=RED, ha="center", va="bottom", cap=_charcap(6.5, 6.4))

    # ablation row — the systems the benchmark compares, same tools/loop/scoring
    _box(ax, 0.3, 7.58, 11.9, 0.42,
         "benchmark systems — same tools · same loop · same scoring:   bare   ·   "
         "soft-gate   ·   self-verify   ·   tool_no_gate   ·   finetuned-ext   ·   "
         "labwright",
         GRAY_LIGHT, GRAY, fs=6.8, pad=0.10)

    # =====================================================================
    # (c)  VERIFIER PIPELINE
    # =====================================================================
    _panel(ax, 0.1, 5.80, 12.4, 7.40,
           "(c)  verifier — re-derives every derived number from the agent's own raw inputs")
    verif = [
        ("1  arithmetic", "re-runs every governing equation\n(checker.py, 8 domains)",
         "error on mismatch"),
        ("2  units & dimensions", "17 alias traps via pint\ndyn/cm2-as-Pa = 10x, "
         "mL/min-as-uL/min...", "error | warning"),
        ("3  physiological sanity", "soft / hard bands\nshear 1e-3-10 Pa soft, "
         "1e-4-50 hard", "error | warning"),
        ("4  safety & compliance", "SafetyConfig: DMSO <0.5%, dose caps,\nvehicle "
         "control, BSL hints, institution", "error | warning"),
        ("5  prose-number gate", "numbers written in the narrative\nmust match a design "
         "value (thresholds skipped)", "warning only"),
    ]
    vx0, vw, vgap, vh = 0.3, 2.28, 0.13, 0.72
    vy = 6.30
    for i, (name, desc, verdict) in enumerate(verif):
        x = vx0 + i * (vw + vgap)
        _box(ax, x, vy, vw, vh, f"{name}\n{desc}", BLUE, BLUE_EDGE, tc="white",
             fs=6.1, pad=0.06)
        _label(ax, x + vw / 2, vy + 0.015, verdict, fs=6.0,
               color=RED if "error on" in verdict else MUT, ha="center",
               va="bottom", weight="bold")
    for i in range(len(verif) - 1):
        _arrow(ax, vx0 + (i + 1) * vw + i * vgap + 0.01, vy + vh / 2,
               vx0 + (i + 1) * (vw + vgap) - 0.03, vy + vh / 2, color=BLUE_EDGE)
    _label(ax, 0.3, 6.02,
           "a design is accepted only when no layer reports an error · prose assertions "
           "are warnings, so an honest design is never blocked",
           fs=6.6, color=INK)

    # =====================================================================
    # (d)  CALCULATOR TOOLBOX
    # =====================================================================
    _panel(ax, 0.1, 2.30, 12.4, 5.80,
           f"(d)  calculator toolbox — {N_TOOLS} tools in {N_CLASSES} classes (pure "
           "functions · pydantic-schema'd · pint units · worked example + unit traps)")
    cols = list(TOOLS.items())
    cw, cgap = 2.2, 0.15
    row1_y, row2_y, ch1, ch2 = 4.62, 3.84, 0.70, 0.62
    for i, (cat, names) in enumerate(cols):
        row = 1 if i < 5 else 0
        y = row1_y if row == 1 else row2_y
        h = ch1 if row == 1 else ch2
        x = 0.3 + (i % 5) * (cw + cgap)
        _box(ax, x, y, cw, h, f"{cat} · {len(names)}\n{' · '.join(names)}",
             GRAY_LIGHT, GRAY, fs=5.5, pad=0.05, line_h=1.12)

    # calc/ modules + the governing equations
    _box(ax, 0.3, 2.72, 11.9, 1.00,
         "calc/ — 10 pure modules; derived numbers are produced HERE and only here:   "
         "microfluidics · cell · culture · spheroid · o2 · barrier · dosing · pk · "
         "stats · units\n"
         "τ = 6μQ/(wh²)   ·   Re = ρūD_h/μ   ·   ΔP = 12μQL/(wh³)   ·   "
         "n = 2(zα+zβ)²σ²/δ²   ·   E = 1 − Cout/Cin   ·   t½ = ln2·V/Cl\n"
         "δ = √(2·D·C0/q)   ·   TEER = (R−Rblank)·A   ·   Papp = J/(A·C0)   ·   "
         "C = N·D·1e4",
         GRAY_LIGHT, GRAY, fs=6.4, pad=0.10)
    _label(ax, 0.3, 2.58,
           "every derived field pinned to one canonical unit by pint (dyn = 1e-5 N); a "
           "new domain = one calc/ module + one register_tool call — not a fork",
           fs=6.2, color=BLUE_EDGE, va="top", cap=_charcap(11.6, 6.2))

    # =====================================================================
    # (e)  INTERNAL COMPONENTS · BENCHMARK SYSTEMS · HONEST BOUNDARY
    # =====================================================================
    _panel(ax, 0.1, 0.05, 12.4, 2.30,
           "(e)  internal components · benchmark systems · honest boundary")
    ey_top = 1.86
    c1 = _charcap(3.85, 6.2)   # column 1 text width (x 0.3 -> 4.15)
    c2 = _charcap(3.6, 6.2)    # column 2 text width (x 4.7 -> 8.30)
    c3 = _charcap(3.55, 6.0)   # column 3 label width

    # column 1 — the agent & brain
    _label(ax, 0.3, ey_top, "the agent & brain", fs=6.8, color=BLUE_EDGE,
           weight="bold")
    _label(ax, 0.3, 1.60, "DesignAgent — ReAct loop; ends only on submit_design; "
           "prose refused; on review_required fixes raw inputs only", fs=6.2, cap=c1)
    _label(ax, 0.3, 1.34, "LLMClient — OpenAI-compatible; deepseek-v4-flash/pro; "
           "T=0.2; thinking off; retry 3x backoff; max_tokens 8192", fs=6.2, cap=c1)
    _label(ax, 0.3, 1.08, "tool_no_gate ablation — same calculators & loop, verifier "
           "off; isolates what the verification layer adds", fs=6.2, cap=c1)
    _label(ax, 0.3, 0.82, "cell_physiology — read-only lookup over the 10-profile "
           "physiology registry", fs=6.2, cap=c1)
    _label(ax, 0.3, 0.56, "verify_published_protocol — same calculators run backwards "
           "over a paper's own inputs (consistent / discrepancy / …)", fs=6.2, cap=c1)
    _label(ax, 0.3, 0.30, "UI (Gradio two-axis badge) · CLI · Colab — all entry "
           "points into the same verified pipeline", fs=6.2, cap=c1)

    # column 2 — fine-tuned extractor subsystem
    _label(ax, 4.7, ey_top, "fine-tuned extractor (single-pass, not an agent)",
           fs=6.8, color=BLUE_EDGE, weight="bold")
    _label(ax, 4.7, 1.60, "Qwen2.5-1.5B-Instruct + LoRA (r=16) tuned on goal -> "
           "raw-inputs; goes through the SAME gate (reject derived -> DesignInput -> "
           "build_design -> verify)", fs=6.2, cap=c2)
    _label(ax, 4.7, 1.28, f"committed eval: {BENCH['extractor']}", fs=6.2,
           color=INK, weight="bold", cap=c2)
    _label(ax, 4.7, 1.02, "the agent and the extractor are the two independent paths "
           "that produce raw inputs; both feed the identical calculators and verifier",
           fs=6.2, cap=c2)
    _label(ax, 4.7, 0.72, "gold sets: 24-reading · 15-blind (8 cold + 5 prompt-backed "
           "+ 2 scenario) · 15-3D-spheroid · 14-plate-culture · 14-perfused-PK",
           fs=6.2, cap=c2)
    _label(ax, 4.7, 0.42, "metrics: usable · self-consistent · hallucination · failure "
           "reason · unit-misread · target-selection · cold-split", fs=6.2, cap=c2)

    # column 3 — honest boundary + committed numbers
    _label(ax, 8.7, ey_top, "honest boundary & committed results", fs=6.8,
           color=BLUE_EDGE, weight="bold")
    _box(ax, 8.7, 0.34, 3.55, 1.20,
         "the verifier checks arithmetic & internal consistency, not target "
         "selection — the UI marks both (green = arithmetic verified / amber = "
         "targets model-proposed)\n"
         f"reading set (24): usable {BENCH['reading_usable']}; hallucination "
         f"{BENCH['reading_halluc']}\n"
         f"{BENCH['reading_ci']}\n"
         f"blind set (15): usable {BENCH['blind_usable']}; hallucination "
         f"{BENCH['blind_halluc']}",
         PALE, GRAY, fs=6.0, pad=0.08, line_h=1.16)
    _label(ax, 8.7, 0.14, "hallucination == 0.000 is attack-tested (prose refusal · "
           "derived-field rejection · tamper detection · prose assertions), not \"zero "
           "by construction\"", fs=6.0, color=MUT, cap=c3, va="bottom")

    # ------------------------------------------------------------------
    # Overlap check before writing
    # ------------------------------------------------------------------
    problems = check_overlaps(fig, ax)
    if problems:
        print("LAYOUT PROBLEMS:")
        for p in problems:
            print(p)
    else:
        print("overlap check: clean")

    out = Path(__file__).resolve().parent
    fig.savefig(out / "fig_architecture.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(out / "fig_architecture.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(out / "fig_architecture.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    print(f"wrote {out / 'fig_architecture.pdf'} and {out / 'fig_architecture.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
