"""Render the protocol information-flow DAG: where every Labwright number comes from.

This is the reviewer-facing "action item" figure — the DAG information-flow
topology of a real verified design. Nothing here is hand-drawn: the field-level
graph in panel (b) is built at runtime from :func:`labwright.sop.provenance.provenance_for`
on a real liver-on-chip plan produced by :func:`labwright.design.submit_design`,
and the numbers on every edge come straight from that provenance.

Panels
------
(a)  Pipeline topology — goal prose -> LLM (raw inputs only) -> calculators ->
     DesignPlan -> verifier gate -> SOP + provenance/ELN. The boundary rule is
     the top line: the LLM never writes a derived number.
(b)  Field-level provenance DAG — every derived field of the plan as a node
     (rounded box), every raw input as a leaf (circle); an edge "X -> Y" means
     "Y's provenance lists X among its inputs". Derived->derived edges are
     labelled with the value that flows (the two- and three-level chains are
     where the calculators compose). Node colour = block; box outline =
     verifier status (green ok / amber warning — this plan has warnings by
     design, from the DMSO-safety and stats-practicality layers).
(c)  Conservation audit — two additive identities recomputed from the plan:
     total_seed_count == seed_per_well × wells, and total cells A+B ==
     ρ·A·wells. These are the same identities the verifier's arithmetic layer
     re-proves; the panel prints pass/fail live.

Usage::

    python paper/fig_protocol_dag.py   # writes paper/fig_protocol_dag.pdf/.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _font import setup_font  # noqa: E402

from labwright.design import submit_design  # noqa: E402

setup_font()

# ---------------------------------------------------------------------------
# Palette — same system identity as fig_architecture / fig_pipeline.
# ---------------------------------------------------------------------------
INK = "#262522"
MUT = "#8a8782"
GRID = "#d9d7d3"
BLUE = "#2E5598"
BLUE_EDGE = "#1f3f70"
BLUE_LIGHT = "#c3d2ec"
GRAY = "#a8a39d"
GRAY_LIGHT = "#e0dcd5"
RED = "#b3261e"
AMBER = "#b98a1e"
GREEN = "#3d7a4f"
PALE = "#f6f4f0"
WHITE = "#ffffff"

# Block accent colours — distinguishable and safe on white.
BLOCK_COLORS = {
    "derived": "#2E5598",     # flow (field prefix "derived.")
    "culture": "#C26A3A",     # ochre
    "dosing": "#6E8A5A",      # sage
    "stats": "#A26A8E",       # violet
    "bioprinting": "#4E7A86", # teal
    "coculture": "#3D7A4F",   # green
    "enzyme": "#7A5A9E",      # purple
    "champ": "#9A6A4A",       # rust
}

#: The example plan the figure is built from — a real, verifier-passed
#: (0 errors; the 4 warnings below are deliberate and honest) liver-on-chip
#: co-culture design. Changing these inputs changes the DAG; nothing is static.
PLAN_INPUT = {
    "goal": "Design a liver-on-chip co-culture dosing study.",
    "rationale": "Rich multi-domain example rendered in the protocol DAG figure.",
    "chip": {"width_um": 800, "height_um": 100, "length_mm": 20},
    "flow": {"flow_rate_uLmin": 1.0, "viscosity_pas": 0.001, "density_kgm3": 1000},
    "culture": {
        "cell_type": "primary human hepatocytes",
        "seeding_density_cells_cm2": 3e5, "plate_format": "6-well", "wells": 6,
        "doubling_time_h": 30, "culture_duration_h": 48,
        "confluent_density_cells_cm2": 1e6,
    },
    "dosing": {"compound": "Compound X", "working_mM": 1, "stock_mM": 10,
               "molecular_weight_g_mol": 250},
    "stats": {"effect_size": 1.5, "std_dev": 0.5, "alpha": 0.05, "power": 0.8},
    "bioprinting": {"nozzle_id": "nozzle_3", "travel_distance_um": 5000,
                    "feed_rate_mm_min": 2.0, "density_g_cm3": 1.0,
                    "footprint_width_um": 1000, "line_pitch_um": 200},
    "coculture": {"cell_type_a": "HUVEC-T1", "cell_type_b": "HepG2",
                  "fraction_a": 0.7, "total_density_cells_cm2": 3e5,
                  "area_cm2": 9.0, "wells": 6},
    "enzyme": {"enzyme": "UGT2B7", "substrate": "morphine", "km_um": 500,
               "s_conc_um": 100, "i_conc_um": 10, "ki_um": 50, "vmax_umol_min": 2.0},
    "champ": {"n_samples": 24, "platform": "450k"},
}

#: The safety warnings are honest output of the verifier's warning layers —
#: keep them so the figure shows a real status colouring, not a scrubbed one.
EXPECTED_WARNINGS = 4


def _fmt(v: float) -> str:
    """Short numeric label: sci below 1e-3 / above 1e6, else 3 sig figs."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, str):
        return v
    if v == 0:
        return "0"
    av = abs(v)
    if av >= 1e6 or av < 1e-3:
        return f"{v:.1e}"
    if av >= 1000:
        return f"{v:,.0f}"
    return f"{v:.3g}"


# ---------------------------------------------------------------------------
# DAG construction — straight from provenance, nothing hand-drawn.
# ---------------------------------------------------------------------------
def build_dag(records: list[dict]):
    """From provenance records build ``(nodes, edges)``.

    ``nodes``: name -> {kind: raw|derived, block, value, unit, status}.
    ``edges``: list of (src, dst, value, unit).
    """
    derived = {r["field"] for r in records}
    nodes: dict[str, dict] = {}
    edges: list[tuple] = []
    for r in records:
        field = r["field"]
        nodes[field] = {
            "kind": "derived", "block": field.split(".")[0],
            "value": r["value"], "unit": r["unit"], "status": r["status"],
            "formula": r["formula"],
        }
        for inp in r["inputs"]:
            edges.append((inp["name"], field, inp["value"], inp["unit"]))
            if inp["name"] not in derived and inp["name"] not in nodes:
                nodes[inp["name"]] = {
                    "kind": "raw", "block": field.split(".")[0],
                    "value": inp["value"], "unit": inp["unit"], "status": "ok",
                    "formula": "",
                }
    return nodes, edges


def _levels(nodes: dict, edges: list) -> dict[str, int]:
    """Longest-path levels: raw leaves at 0, derived at 1+max(input levels)."""
    level = {n: 0 for n, v in nodes.items() if v["kind"] == "raw"}
    indeg = {n: 0 for n in nodes}
    kids: dict[str, list[str]] = {n: [] for n in nodes}
    for src, dst, *_ in edges:
        if src in nodes:
            kids[src].append(dst)
            indeg[dst] += 1
    # Kahn from the raw leaves.
    queue = [n for n in level]
    while queue:
        n = queue.pop(0)
        for k in kids[n]:
            level[k] = max(level.get(k, 0), level[n] + 1)
            indeg[k] -= 1
            if indeg[k] == 0:
                queue.append(k)
    return level


# ---------------------------------------------------------------------------
# Canvas helpers (same style contract as fig_architecture).
# ---------------------------------------------------------------------------
def _box(ax, x, y, w, h, text, fc, ec, tc=WHITE, fs=7.0, weight="normal",
         pad=0.10, zorder=3, line_h=1.2):
    ax.add_patch(FancyBboxPatch(
        (x + pad, y + pad), w - 2 * pad, h - 2 * pad,
        boxstyle=f"round,pad={pad},rounding_size=0.08",
        fc=fc, ec=ec, linewidth=1.1, zorder=zorder))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=zorder + 1, fontweight=weight,
            linespacing=line_h)


def _arrow(ax, x1, y1, x2, y2, color=MUT, style="-|>", lw=1.2, zorder=2, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=11, color=color, linewidth=lw,
                                 zorder=zorder, linestyle=ls))


def _label(ax, x, y, text, fs=6.4, color=MUT, ha="left", va="bottom", zorder=2):
    ax.text(x, y, text, fontsize=fs, color=color, ha=ha, va=va, zorder=zorder)


def _off(ax) -> None:
    """Turn an axes off and drop its phantom tick-label texts.

    ``axis("off")`` hides ticks but leaves the tick-label artists with their
    text in place; the overlap census counts them as if visible, so an axes that
    shows no ticks must also carry no tick-label text.
    """
    ax.axis("off")
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def check_overlaps(fig) -> list[str]:
    """Text-text overlap census over every artist the renderer inspects."""
    fig.canvas.draw()
    rnd = fig.canvas.get_renderer()
    boxes: list[tuple[str, float, float, float, float]] = []
    for art in list(fig.texts) + [a for ax in fig.axes for a in ax.texts]:
        bb = art.get_window_extent(rnd)
        boxes.append((art.get_text().replace("\n", "\\n")[:40], bb.x0, bb.y0, bb.x1, bb.y1))
    problems: list[str] = []
    for a in range(len(boxes)):
        for b in range(a + 1, len(boxes)):
            ax0, ay0, ax1, ay1 = boxes[a][1:]
            bx0, by0, bx1, by1 = boxes[b][1:]
            ox = max(0.0, min(ax1, bx1) - max(ax0, bx0))
            oy = max(0.0, min(ay1, by1) - max(ay0, by0))
            if ox > 2 and oy > 2:
                problems.append(f"  {boxes[a][0]!r} OVERLAPS {boxes[b][0]!r} ({ox:.0f}x{oy:.0f}px)")
    return problems


def _render_block_dag(ax, block: str, nodes: dict, edges: list):
    """Draw one block's field-level DAG into an axes in [0,1]^2 space.

    Only this block's derived nodes are drawn; the raw leaves drawn are exactly
    those that feed them (raw inputs shared with other blocks — width_um, wells,
    plate_format — appear here in their feeding colour).
    """
    derived = {d for s, d, *_ in edges if nodes[d]["block"] == block}
    raws_used = {s for s, d, *_ in edges if d in derived and s in nodes and nodes[s]["kind"] == "raw"}
    keep = derived | raws_used
    sub_edges = [(s, d, v, u) for s, d, v, u in edges if d in keep and s in keep]
    sub_nodes = {n: nodes[n] for n in keep}
    level = _levels(sub_nodes, sub_edges)

    # Position: raw leaves at the bottom row, derived fields stacked above by
    # longest-path level. Everything stays inside the cell so adjacent cells
    # never collide.
    maxlvl = max(level.values(), default=0)
    lvl_order: dict[int, list[str]] = {}
    for n, l in level.items():
        lvl_order.setdefault(l, []).append(n)
    pos: dict[str, tuple[float, float]] = {}
    for l, names in lvl_order.items():
        for i, n in enumerate(names):
            x = 0.5 if len(names) == 1 else 0.08 + 0.84 * i / (len(names) - 1)
            y = 0.12 if l == 0 else 0.30 + 0.50 * (l - 1) / max(maxlvl - 1, 1)
            pos[n] = (x, y)

    # Edges first (behind nodes).
    for s, d, v, u in sub_edges:
        (x1, y1), (x2, y2) = pos[s], pos[d]
        mid = (y1 + y2) / 2
        ax.plot([x1, x1, x2], [y1, mid, y2], color=GRID, lw=1.0, zorder=1)
        # derived->derived edges carry the flowing value label
        if sub_nodes[s]["kind"] == "derived":
            lx, ly = (x1 + x2) / 2, (y1 + y2) / 2 + 0.045
            ax.text(lx, ly, f"{_fmt(v)} {u}", ha="center", va="bottom",
                    fontsize=4.6, color=MUT, zorder=2)

    # Raw leaves (circles), then derived nodes (rounded boxes).
    for n, nd in sub_nodes.items():
        x, y = pos[n]
        if nd["kind"] == "raw":
            ax.add_patch(Circle((x, y), 0.042, fc=BLOCK_COLORS.get(nd["block"], GRAY),
                                ec="#74706A", lw=0.8, zorder=3))
            ax.text(x, y - 0.085, f"{n.split('.')[-1]}\n{_fmt(nd['value'])} {nd['unit']}",
                    ha="center", va="top", fontsize=4.6, color=INK, zorder=4)
        else:
            edge_c = GREEN if nd["status"] == "ok" else AMBER
            fc = BLOCK_COLORS.get(nd["block"], GRAY_LIGHT)
            ax.add_patch(FancyBboxPatch(
                (x - 0.145, y - 0.035), 0.29, 0.085,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                fc=fc, ec=edge_c, lw=1.0, zorder=3))
            ax.text(x, y + 0.002,
                    f"{n.split('.')[-1]} = {_fmt(nd['value'])} {nd['unit']}",
                    ha="center", va="center", fontsize=4.8, color=WHITE, zorder=4)


def main() -> int:
    # Build the real plan + provenance once.
    result = submit_design(PLAN_INPUT)
    issues = result["verification"]
    records = result["provenance"]
    nodes, edges = build_dag(records)

    n_err = sum(1 for i in issues if i["level"] == "error")
    n_warn = sum(1 for i in issues if i["level"] == "warning")
    assert n_err == 0, f"DAG example plan must have 0 errors, got {n_err}"
    # Keep warnings — they are honest verifier output (DMSO safety, stats n).

    # Conservation identities (the verifier's arithmetic layer re-proves these).
    prov = {r["field"]: r for r in records}
    cons = {
        "cells": (prov["coculture.total_cells_a"]["value"]
                  + prov["coculture.total_cells_b"]["value"],
                  PLAN_INPUT["coculture"]["total_density_cells_cm2"]
                  * PLAN_INPUT["coculture"]["area_cm2"]
                  * PLAN_INPUT["coculture"]["wells"]),
        "seed": (prov["culture.total_seed_count"]["value"],
                 prov["culture.seed_per_well"]["value"] * PLAN_INPUT["culture"]["wells"]),
    }
    cons_pass = all(abs(a - b) < 1e-6 * max(1.0, abs(a), abs(b)) for a, b in cons.values())

    fig, _ax0 = plt.subplots(figsize=(15.0, 17.0))
    fig.delaxes(_ax0)
    fig.patch.set_facecolor("white")

    # =====================================================================
    # (a)  PIPELINE TOPOLOGY
    # =====================================================================
    ga = fig.add_axes([0.03, 0.93, 0.94, 0.055])
    ga.set_xlim(0, 15.0); ga.set_ylim(0, 1); _off(ga)
    ga.text(0, 0.96, "(a)  pipeline topology — the boundary rule: the LLM proposes RAW inputs; "
                     "deterministic calculators compute every number; the verifier re-proves",
            fontsize=8.4, color=BLUE_EDGE, weight="bold", va="top")
    flow = [
        ("goal prose", "wet-lab goal in\nnatural language", "raw"),
        ("LLM raw inputs", "ReAct agent proposes\nRAW inputs only", "blue"),
        ("submit_design", "hard gate:\nderived fields rejected", "blue"),
        ("calculators", "pure calc/ per block\nevery number computed here", "blue"),
        ("DesignPlan", "design JSON + all\nderived fields", "blue"),
        ("verifier gate", "re-derives + sanity bands\nok -> accept, else fix raws", "blue"),
        ("SOP + provenance", "protocol, ELN/LIMS record\nformula + inputs + status", "blue"),
    ]
    bx0, bw, bgap, bh = 0.12, 1.86, 0.20, 0.62
    for i, (head, desc, kind) in enumerate(flow):
        x = bx0 + i * (bw + bgap)
        fc = GRAY_LIGHT if kind == "raw" else BLUE
        ec = GRAY if kind == "raw" else BLUE_EDGE
        _box(ga, x, 0.10, bw, bh, f"{head}\n{desc}", fc, ec, fs=6.3, pad=0.05)
        if i < len(flow) - 1:
            _arrow(ga, x + bw + 0.012, 0.42, x + bw + bgap - 0.012, 0.42,
                   color=BLUE_EDGE, lw=1.4)

    # =====================================================================
    # (b)  FIELD-LEVEL PROVENANCE DAG — one axes per block
    # =====================================================================
    gb = fig.add_axes([0.03, 0.315, 0.94, 0.60])
    gb.set_xlim(0, 15.0); gb.set_ylim(0, 1); _off(gb)
    gb.text(0.1, 0.985, "(b)  field-level provenance DAG — built from provenance_for() on the plan "
                     "above; every edge is a real input->field dependency, edges between derived "
                     "fields carry the value that flows",
            fontsize=8.4, color=BLUE_EDGE, weight="bold", va="top")

    # 2x2 grid of the four deep-chain domains (the multi-level calculators
    # where derived values compose into derived values). Flat one-step blocks
    # (flow/dosing/stats/champ) are summarised in a line below — their full
    # provenance records are in the same traceability logs.
    DISPLAY = {
        "derived": "flow (microfluidics)", "culture": "culture", "dosing": "dosing",
        "stats": "stats (power)", "bioprinting": "bioprinting", "coculture": "coculture",
        "enzyme": "enzyme (Cheng–Prusoff)", "champ": "champ (methylation arrays)",
    }
    deep_blocks = ["culture", "bioprinting", "coculture", "enzyme"]
    cell_w, cell_h, hgap, vgap = 7.2, 0.42, 0.20, 0.04
    for i, block in enumerate(deep_blocks):
        col, row = i % 2, i // 2
        x0 = 0.10 + col * (cell_w + hgap)
        y0 = 0.54 - row * (cell_h + vgap)
        ax = fig.add_axes([0.03 + 0.94 * x0 / 15.0, 0.315 + 0.60 * y0,
                           0.94 * cell_w / 15.0, 0.60 * cell_h])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); _off(ax)
        _render_block_dag(ax, block, nodes, edges)
        ax.text(0.5, 0.945, DISPLAY[block], ha="center", va="bottom", fontsize=7.2,
                color=BLOCK_COLORS[block], weight="bold")

    flat = ("flat one-step blocks (same provenance machinery, single level):   flow → "
            "shear_pa · reynolds · pressure_drop_pa · residence_time_s · channel_volume_ul · "
            "mean_velocity_mms    dosing → dmso_fraction_vv    stats → n_per_group    "
            "champ → n_arrays · n_chips")
    gb.text(0.10, 0.005, flat, fontsize=6.2, color=MUT, va="bottom", ha="left")
    gb.text(0.10, 0.51, "●  raw input (model may propose)      "
                        "■  derived (computers-only, model never writes)      "
                        "—  green box edge = verifier ok   ·   amber = warning",
            fontsize=6.6, color=INK, va="top")

    # =====================================================================
    # (c)  CONSERVATION AUDIT
    # =====================================================================
    gc = fig.add_axes([0.03, 0.035, 0.94, 0.24])
    gc.set_xlim(0, 15.0); gc.set_ylim(0, 1); _off(gc)
    gc.text(0, 0.96, "(c)  conservation audit — additive identities recomputed from the same plan "
                    "(the verifier's arithmetic layer re-proves exactly these)",
            fontsize=8.4, color=BLUE_EDGE, weight="bold", va="top")
    lines = [
        f"cells  N_A + N_B  =  {_fmt(cons['cells'][0])}   vs   ρ·A·wells = {_fmt(cons['cells'][1])}   "
        f"{'✓ in = out' if abs(cons['cells'][0]-cons['cells'][1])<1e-6*max(1,cons['cells'][0],cons['cells'][1]) else '✗ MISMATCH'}",
        f"seed   N_well × wells  =  {_fmt(cons['seed'][0])}   vs   seed_per_well × 6 = {_fmt(cons['seed'][1])}   "
        f"{'✓ in = out' if abs(cons['seed'][0]-cons['seed'][1])<1e-6*max(1,cons['seed'][0],cons['seed'][1]) else '✗ MISMATCH'}",
        f"plan status: {result['status']}  ({n_err} errors · {n_warn} warnings) — warnings are the DMSO-safety and "
        f"stats-n layers, intentionally kept so the status colouring is real",
    ]
    for i, line in enumerate(lines):
        gc.text(0.03, 0.66 - i * 0.20, line, fontsize=7.4, color=INK, va="top",
                family="monospace")

    # ------------------------------------------------------------------
    # Overlap census before writing
    # ------------------------------------------------------------------
    problems = check_overlaps(fig)
    if problems:
        print(f"LAYOUT PROBLEMS ({len(problems)}):")
        for p in problems:
            print(p)
    else:
        print("overlap check: clean")

    out = Path(__file__).resolve().parent
    fig.savefig(out / "fig_protocol_dag.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(out / "fig_protocol_dag.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(out / "fig_protocol_dag.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    print(f"wrote {out / 'fig_protocol_dag.pdf'} and {out / 'fig_protocol_dag.png'}")
    print(f"({len(nodes)} nodes · {len(edges)} edges from provenance; conservation pass={cons_pass})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
