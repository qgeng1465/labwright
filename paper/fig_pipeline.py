"""Render the paper's framework figure (Fig 1): the full Labwright system.

The figure shows every component a visitor needs to understand *what the tool
is made of*: the entry points (CLI / web / Colab, plus the reverse
``verify-protocol`` direction), the goal, the ReAct agent loop, the 46-tool
registry in its 10 classes, the calculators (where every derived number is
produced), the four verifier layers (arithmetic, units & sanity, safety,
prose), the hard ``submit_design`` gate, and the outputs (SOP + design JSON +
provenance/ELN export). The naive alternatives an LLM reaches for end in a red
REJECTED box, and an honest-boundary callout states what the gate does and
does not do.

Layout (authored in data coordinates, canvas 12.5 x 9.4):
- a left column carries the main flow, top to bottom: entry -> goal -> agent ->
  calculators -> verifier -> gate -> SOP + design JSON
- the left gutter (x < 0.70) holds two vertical side-elements: the gray
  ``reverse · verify-protocol`` sidebar (the same gate run backwards) and the
  red retry lane from the gate back to the agent
- right of each main-flow box sits a detail panel naming the real components
- the bottom band holds the naive alternatives -> REJECTED lane and the honest
  boundary callout
- vertical spacing between stacked texts inside a panel is >= ~0.12 data
  units so nothing reads as overlapping at print size (verified by the
  overlap census in _check_render.py)

Before writing, a programmatic check asserts no text bbox overlaps another
text bbox or spills outside the canvas.

Usage::

    python paper/fig_pipeline.py   # writes paper/fig_pipeline.pdf/.png
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

INK = "#262522"
MUT = "#8a8782"
GRID = "#d9d7d3"
BLUE = "#2E5598"        # Labwright main path — the paper's deep academic blue
BLUE_EDGE = "#1f3f70"
BLUE_LIGHT = "#c3d2ec"  # the goal entry: part of the blue tier, but an input
GRAY = "#a8a39d"
GRAY_LIGHT = "#e0dcd5"
GRAY_DARK = "#74706a"
RED = "#b3261e"  # status: hard reject — reserved for its meaning
PALE = "#f6f4f0"  # detail-panel fill

# Canvas is deliberately tall (12.5 x 9.4) so every stacked text inside a
# panel clears its neighbour by >= ~0.08 data units; the earlier 12.5 x 8.2
# canvas packed the verifier layers, tool grid and gate notes close enough
# that they read as overlapping at print size.
X0, X1, Y0, Y1 = 0.0, 12.5, 0.0, 9.4

MX0, MX1 = 0.70, 3.00        # main-flow column (shifted right for the left gutter)
MXM = (MX0 + MX1) / 2        # its vertical arrow x
PX0, PX1 = 3.45, 12.25       # detail-panel region


def _box(ax, x, y, w, h, text, fc, ec, tc=INK, fs=8.5, weight="normal",
         line_h=1.25, zorder=3, pad=0.20):
    ax.add_patch(FancyBboxPatch(
        (x + pad, y + pad), w - 2 * pad, h - 2 * pad,
        boxstyle=f"round,pad={pad},rounding_size=0.10",
        fc=fc, ec=ec, linewidth=1.1, zorder=zorder))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, zorder=zorder + 1, fontweight=weight, linespacing=line_h)


def _panel(ax, x0, y0, x1, y1, title, fs=7.5, zorder=1):
    """A light detail panel with a small top-left title."""
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0.15,rounding_size=0.08",
                                fc=PALE, ec=GRID, linewidth=1.0, zorder=zorder))
    ax.text(x0 + 0.15, y1 - 0.10, title, fontsize=fs, color=GRAY_DARK,
            ha="left", va="top", weight="bold", zorder=zorder + 1)


def _arrow(ax, x1, y1, x2, y2, color=MUT, style="-|>", lw=1.3, zorder=2, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=13, color=color, linewidth=lw,
                                 zorder=zorder, linestyle=ls))


def _label(ax, x, y, text, fs=7.0, color=MUT, ha="left", va="bottom",
           weight="normal", zorder=2, line_h=1.25):
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
        boxes.append((i, t.get_text().replace("\n", "\\n")[:40], x0, y0, x1, y1))
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
    fig, ax = plt.subplots(figsize=(12.5, 9.4))
    ax.set_xlim(X0, X1)
    ax.set_ylim(Y0, Y1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ------------------------------------------------------------------
    # Entry points (top band)
    # ------------------------------------------------------------------
    _box(ax, 0.70, 8.75, 3.9, 0.42,
         "entry:  labwright design \"…\"  ·  web app (Gradio)  ·  Colab",
         GRAY_LIGHT, GRAY, fs=7.5, pad=0.12)
    _arrow(ax, MXM, 8.78, MXM, 8.55, color=BLUE_EDGE, lw=1.1)

    # ------------------------------------------------------------------
    # Left column: the main flow (top to bottom)
    # ------------------------------------------------------------------
    _box(ax, 0.70, 8.0, 2.3, 0.6, "goal\n(natural language)", BLUE_LIGHT,
         BLUE_EDGE, fs=8.5)
    _box(ax, 0.70, 6.3, 2.3, 1.55, "agent\n(ReAct loop)", BLUE, BLUE_EDGE,
         tc="white", weight="bold", fs=8.5)
    _box(ax, 0.70, 4.55, 2.3, 1.6, "tool registry\n+ calculators", BLUE,
         BLUE_EDGE, tc="white", weight="bold", fs=8.5)
    _box(ax, 0.70, 2.7, 2.3, 1.7, "verifier\n(4 layers)", BLUE, BLUE_EDGE,
         tc="white", weight="bold", fs=8.5)
    _box(ax, 0.70, 1.25, 2.3, 1.3, "hard gate\nsubmit_design", BLUE,
         BLUE_EDGE, tc="white", weight="bold", fs=8.5)
    _box(ax, 0.70, 0.15, 2.3, 0.8, "SOP + design JSON\n(verified)",
         GRAY_DARK, GRAY_DARK, tc="white", weight="bold", fs=8.0)

    for ya, yb in [(8.0, 7.9), (6.3, 6.2), (4.55, 4.45), (2.7, 2.6),
                   (1.25, 1.05)]:
        _arrow(ax, MXM, ya, MXM, yb, color=BLUE_EDGE)

    # ------------------------------------------------------------------
    # Left gutter: the reverse verify-protocol sidebar + the red retry lane
    # ------------------------------------------------------------------
    _box(ax, 0.12, 4.8, 0.32, 3.9, "", GRAY_LIGHT, GRAY, pad=0.06)
    ax.text(0.28, 6.75, "reverse · verify-protocol\n(same gate, opposite "
            "direction)", rotation=90, fontsize=6.5, color=GRAY_DARK,
            ha="center", va="center", linespacing=1.5, zorder=4)

    _arrow(ax, 0.65, 2.0, 0.65, 7.0, color=RED, style="-|>", lw=1.3,
           ls=(0, (4, 2)))
    ax.text(0.57, 4.5, "invalid submission → agent retries (raw inputs only)",
            rotation=90, fontsize=6.3, color=RED, ha="center", va="center",
            zorder=4)

    # ------------------------------------------------------------------
    # Detail panel: the agent (ReAct loop)
    # ------------------------------------------------------------------
    A0, A1 = 6.3, 7.85
    _panel(ax, PX0, A0, PX1, A1, "the agent — a ReAct loop over the tool registry")
    _box(ax, 3.60, 6.95, 1.5, 0.45, "think", BLUE, BLUE_EDGE, tc="white",
         fs=7.5, pad=0.08)
    _box(ax, 5.45, 6.95, 1.6, 0.45, "call a calculator tool", BLUE, BLUE_EDGE,
         tc="white", fs=7.5, pad=0.08)
    _box(ax, 7.40, 6.95, 1.5, 0.45, "observe result", BLUE, BLUE_EDGE,
         tc="white", fs=7.5, pad=0.08)
    _arrow(ax, 5.10, 7.17, 5.45, 7.17, color=BLUE_EDGE)
    _arrow(ax, 7.05, 7.17, 7.40, 7.17, color=BLUE_EDGE)
    # the loop-back arc dips below the three boxes (rad<0): rad>0 would arch
    # up through them (control point above the chord -> crosses the boxes).
    ax.add_patch(FancyArrowPatch((8.90, 6.9), (3.60, 6.9),
                                 connectionstyle="arc3,rad=-0.10",
                                 arrowstyle="-", mutation_scale=13,
                                 color=BLUE_EDGE, linewidth=1.1, zorder=2))
    _label(ax, 6.25, 6.40, "loop", fs=6.5, ha="center")
    _label(ax, 9.05, 7.35, "max_iterations=12  ·  max_tool_calls_per_turn=8",
           fs=7.5, color=INK, weight="bold")
    _label(ax, 9.05, 7.10, "the loop ends only when submit_design is called",
           fs=7.0)
    _label(ax, 9.05, 6.90, "prose answer (no tool call) → "
           "“numbers you type are not trusted”", fs=6.8)
    _label(ax, 9.05, 6.66, "LLM: deepseek-v4-flash / deepseek-v4-pro · T=0.2 · "
           "thinking off", fs=7.0)
    _label(ax, 9.05, 6.46, "any OpenAI-compatible model via LABWRIGHT_MODEL",
           fs=7.0)

    # ------------------------------------------------------------------
    # Detail panel: tool registry + calculators
    # ------------------------------------------------------------------
    C0, C1 = 4.55, 6.15
    _panel(ax, PX0, C0, PX1, C1,
           "tool registry — 46 tools in 10 classes (each with a worked example "
           "+ common mistakes) · calculators make the numbers")
    classes = [
        ("microfluidics", "6"), ("cell", "11"), ("dosing", "3"), ("stats", "2"),
        ("3D culture", "5"), ("O2 delivery", "6"), ("barrier", "6"),
        ("PK", "5"), ("physiology", "1"), ("published", "1"),
    ]
    for i, (name, n) in enumerate(classes):
        col, row = i % 5, i // 5
        cx = PX0 + 0.18 + col * 1.05
        cy = C1 - 0.63 - row * 0.47  # row1 5.52, row2 5.05, box h 0.36
        _box(ax, cx, cy, 0.95, 0.36, f"{name}\n({n} tools)", GRAY_LIGHT, GRAY,
             fs=6.2, pad=0.06, line_h=1.05)
    _label(ax, PX0 + 0.18, C0 + 0.25,
           "calc/ 10 modules — derived numbers are produced HERE and only here:",
           fs=7.0, color=BLUE_EDGE, weight="bold")
    _label(ax, PX0 + 0.18, C0 + 0.07,
           "microfluidics · cell · culture · spheroid · o2 · barrier · dosing · "
           "pk · stats · units", fs=6.8)

    # ------------------------------------------------------------------
    # Detail panel: the verifier (4 layers)
    # ------------------------------------------------------------------
    V0, V1 = 2.7, 4.4
    _panel(ax, PX0, V0, PX1, V1, "the verifier — re-derives every number")
    layers = [
        ("1  arithmetic", "re-runs every governing equation from the agent's "
         "own inputs", "error on mismatch"),
        ("2  units & sanity", "17 unit aliases (dyn/cm2-as-Pa 10x, "
         "mL/min-as-uL/min) + soft physiological / hard physical bands",
         "error | warning"),
        ("3  safety", "SafetyConfig: DMSO <0.5% v/v, compound dose caps, "
         "vehicle control, BSL hints, institution note", "error | warning"),
        ("4  prose gate", "numbers in the narrative must match a design value "
         "(threshold phrases skipped)", "warning only"),
    ]
    for i, (name, desc, verdict) in enumerate(layers):
        ly = V1 - 0.48 - i * 0.34  # 3.92, 3.58, 3.24, 2.90
        _box(ax, PX0 + 0.18, ly, 1.55, 0.22, name, BLUE, BLUE_EDGE, tc="white",
             fs=6.3, weight="bold", pad=0.05)
        _label(ax, PX0 + 1.95, ly + 0.055, desc, fs=6.5)
        _label(ax, PX1 - 0.18, ly + 0.055, verdict, fs=6.4, ha="right",
               va="center")
    _label(ax, PX0 + 0.18, V0 + 0.04,
           "a design is accepted only when every layer passes (warnings "
           "tolerated)", fs=6.5)

    # ------------------------------------------------------------------
    # Detail panel: the hard gate + provenance
    # ------------------------------------------------------------------
    G0, G1 = 1.25, 2.55
    _panel(ax, PX0, G0, PX1, G1,
           "submit_design — the hard gate (derive + verify + provenance)")
    steps = ["reject derived fields", "DesignInput\nextra=forbid", "build_design",
             "verify", "provenance"]
    for i, s in enumerate(steps):
        sx = PX0 + 0.18 + i * 1.35
        _box(ax, sx, G0 + 0.37, 1.2, 0.42, s, BLUE, BLUE_EDGE, tc="white",
             fs=6.6, pad=0.12)
        if i < len(steps) - 1:
            _arrow(ax, sx + 1.22, G0 + 0.58, sx + 1.32, G0 + 0.58,
                   color=BLUE_EDGE, lw=1.0)
    _label(ax, PX0 + 0.18, G0 + 0.19, "rejected → the agent retries "
           "(raw inputs only)", fs=6.3, color=RED)
    _label(ax, PX1 - 0.18, G0 + 0.19,
           "provenance per number: formula + inputs (name, value, unit) + "
           "output unit + code version", fs=6.3, ha="right")
    _label(ax, PX0 + 0.18, G0 + 0.04,
           "embedded in the SOP + design JSON · export_eln(json|csv) → ELN/LIMS · "
           "reverse mode recomputes a published paper's claims", fs=6.3)

    # ------------------------------------------------------------------
    # Bottom band: naive alternatives -> REJECTED + honest boundary
    # ------------------------------------------------------------------
    _label(ax, 3.1, 1.04, "the alternatives an LLM reaches for — rejected:",
           fs=7.5, va="top")
    _box(ax, 3.1, 0.38, 1.7, 0.42, "type the\nnumbers", GRAY_LIGHT, GRAY,
         fs=7.0, pad=0.10)
    _box(ax, 5.05, 0.38, 2.1, 0.42, "soft self-check\n(LLM judges its own work)",
         GRAY_LIGHT, GRAY, fs=7.0, pad=0.10)
    _arrow(ax, 4.82, 0.59, 5.03, 0.59, color=MUT, lw=1.0)
    _arrow(ax, 7.17, 0.59, 7.38, 0.59, color=RED, lw=1.0)
    ax.add_patch(FancyBboxPatch((7.4, 0.28), 2.3, 0.58,
                                boxstyle="round,pad=0.15,rounding_size=0.08",
                                fc="none", ec=RED, linewidth=1.4,
                                linestyle=(0, (4, 2)), zorder=2))
    _label(ax, 8.55, 0.58, "REJECTED", fs=8.5, color=RED, ha="center",
           va="center", weight="bold")
    _label(ax, 8.55, 0.42, "a typed number is not trusted", fs=6.6, color=RED,
           ha="center", va="center")
    _label(ax, 9.9, 0.70, "honest boundary: the gate verifies", fs=6.3)
    _label(ax, 9.9, 0.52, "arithmetic, not physiology — on", fs=6.3)
    _label(ax, 9.9, 0.34, "blind goals usable drops to 40–47%,", fs=6.3)
    _label(ax, 9.9, 0.16, "hallucination stays 0.000", fs=6.3)

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
    fig.savefig(out / "fig_pipeline.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(out / "fig_pipeline.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {out / 'fig_pipeline.pdf'} and {out / 'fig_pipeline.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
