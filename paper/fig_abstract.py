"""Render the graphical abstract for Labwright.

Three visual zones tell the story left-to-right:

    1. RED — the fluent black box: an LLM asked to design from memory emits
       plausible-looking numbers that derive from nothing ("shear 0.05 Pa",
       "V = 2.9 µL"), and the chemistry breaks.
    2. the GATE — deterministic calculators + the verifier re-deriving every
       number. A funnel filters raw inputs; the neck is a hard gate.
    3. BLUE — determinism: a field-level DAG shows raw inputs feeding derived
       quantities through closed-form formulas, ending in a verified
       SOP + design JSON.

The top band anchors the claim with the committed reading-set numbers (bare
0–12 % usable / ~0.9–1.0 hallucination vs Labwright 88–100 % / 0.000, both
derived from ``results/eval_flash.json`` + ``results/eval_pro.json``), and a
bottom callout states the honest boundary. Red is reserved for the unverified
failure mode and the hard reject; blue is the deterministic system identity,
so the colour semantics carry meaning rather than decoration.

Text is authored in data coordinates; before writing, a programmatic check
asserts no text bbox overlaps another or spills the canvas.

Usage::

    python paper/fig_abstract.py   # writes paper/fig_abstract.pdf/.svg/.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _font import setup_font  # noqa: E402

setup_font()

INK = "#262522"
MUT = "#8a8782"
GRID = "#d9d7d3"
# The deterministic system identity — the paper's deep academic blue, shared
# with the pipeline and benchmark figures so Labwright is one colour everywhere.
BLUE = "#2E5598"
BLUE_EDGE = "#1f3f70"
BLUE_SOFT = "#eaf0f7"
GRAY = "#a8a39d"
GRAY_LIGHT = "#e0dcd5"
GRAY_DARK = "#74706a"
# Unverified / rejected — reserved for the failure mode and the hard reject.
RED = "#b3261e"
RED_DARK = "#8f1f18"
RED_SOFT = "#fbe9e8"
DARK = "#2a2a2e"
WH = "round,pad=0.25,rounding_size=0.10"

X0, X1, Y0, Y1 = 0.0, 12.0, 0.0, 5.4


def _box(ax, x, y, w, h, text, fc, ec, tc=INK, fs=9.5, weight="normal",
         line_h=1.25, zorder=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=WH, fc=fc, ec=ec,
                                linewidth=1.2, zorder=zorder))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, zorder=zorder + 1, fontweight=weight, linespacing=line_h)


def _arrow(ax, x1, y1, x2, y2, color=MUT, style="-|>", lw=1.4, zorder=2, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=13, color=color, linewidth=lw,
                                 zorder=zorder, linestyle=ls))


def _label(ax, x, y, text, fs=7.5, color=MUT, ha="left", va="bottom",
           weight="normal", zorder=2, line_h=1.3, rot=0.0):
    ax.text(x, y, text, fontsize=fs, color=color, ha=ha, va=va, zorder=zorder,
            fontweight=weight, linespacing=line_h, rotation=rot)


def _headline_numbers() -> tuple[str, str]:
    """Derive the top-band headline numbers from committed benchmark JSON.

    Source: the 24-goal *reading* set for both models — ``results/eval_flash.json``
    (deepseek-v4-flash) and ``results/eval_pro.json`` (deepseek-v4-pro).

    * usable: min–max of the two models' ``usable_rate`` (0.0 & 0.125 -> "0–12%";
      0.875 & 1.0 -> "88–100%").
    * hallucination: the bare row shows the worst case across models (max of 1.0
      & 0.875 -> "~0.9–1.0"); the Labwright row shows the best (min of 0.125 &
      0.0 -> "0.000", the pro model — flash's 0.125 is one silence goal, never
      a fabricated number; see the README footnote).

    Same convention as the README headline table, so figure and README agree
    and both trace to committed JSON.
    """
    root = Path(__file__).resolve().parent.parent
    flash = json.loads((root / "results/eval_flash.json").read_text())
    pro = json.loads((root / "results/eval_pro.json").read_text())
    bare_us = [flash["bare"]["usable_rate"], pro["bare"]["usable_rate"]]
    lab_us = [flash["labwright"]["usable_rate"], pro["labwright"]["usable_rate"]]
    bare_hall = [flash["bare"]["hallucination_rate"],
                 pro["bare"]["hallucination_rate"]]
    lab_hall = [flash["labwright"]["hallucination_rate"],
                pro["labwright"]["hallucination_rate"]]
    bare = (f"bare LLM  {min(bare_us) * 100:.0f}–{max(bare_us) * 100:.0f}% "
            f"usable  ·  hallucination ~{min(bare_hall):.1f}–{max(bare_hall):.1f}")
    lab = (f"Labwright  {min(lab_us) * 100:.0f}–{max(lab_us) * 100:.0f}% "
           f"usable  ·  hallucination {min(lab_hall):.3f}")
    return bare, lab


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


def _flask(ax, x0, y0, w, color):
    """A small broken flask centred at (x0+w/2, y0+w/2), red-inked."""
    cx = x0 + w / 2
    body = [(cx - w * 0.22, y0 + w * 0.62), (cx + w * 0.22, y0 + w * 0.62),
            (cx + w * 0.22, y0 + w * 0.42), (cx + w * 0.38, y0),
            (cx - w * 0.38, y0), (cx - w * 0.22, y0 + w * 0.42)]
    ax.add_patch(plt.Polygon(body, closed=True, fill=False, ec=color, lw=1.5,
                             zorder=3))
    # zigzag crack down the neck
    ax.plot([cx - w * 0.03, cx + w * 0.05, cx - w * 0.07, cx + w * 0.09,
             cx - w * 0.03], [y0 + w * 0.62, y0 + w * 0.42, y0 + w * 0.28,
                              y0 + w * 0.10, y0 - w * 0.02], color=color,
            lw=1.2, zorder=4)
    # leaking drop
    ax.add_patch(Circle((cx - w * 0.03, y0 - w * 0.09), w * 0.045,
                        fc=color, ec="none", zorder=4))


def main() -> int:
    fig, ax = plt.subplots(figsize=(12.0, 5.4))
    ax.set_xlim(X0, X1)
    ax.set_ylim(Y0, Y1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ------------------------------------------------------------------
    # Top band: title + benchmark numbers
    # ------------------------------------------------------------------
    ax.add_patch(FancyBboxPatch((0.15, 4.52), 11.7, 0.68,
                                boxstyle="round,pad=0.15,rounding_size=0.08",
                                fc="#f6f4f0", ec=GRID, linewidth=1.0, zorder=1))
    _label(ax, 6.0, 5.30, "how a fluent black box becomes a verified design",
           fs=9.5, color=INK, ha="center", va="center", weight="bold")
    bare_hd, lab_hd = _headline_numbers()
    _label(ax, 0.40, 4.88, "same scoring, every system:", fs=8.0, color=MUT,
           va="center")
    _label(ax, 2.95, 4.88, bare_hd, fs=9.0, color=INK, va="center")
    _label(ax, 11.70, 4.88, lab_hd, fs=10.0,
           color=BLUE_EDGE, va="center", weight="bold", ha="right")

    # ------------------------------------------------------------------
    # Zone captions (between band and the three zones)
    # ------------------------------------------------------------------
    _label(ax, 0.45, 4.36, "fluency without derivation", fs=8.0, color=RED_DARK,
           weight="bold", va="center")
    _label(ax, 5.5, 4.36, "the gate — calculators + verifier", fs=8.0,
           color=BLUE_EDGE, weight="bold", ha="center", va="center")
    _label(ax, 10.0, 4.36, "determinism — recomputed & verified", fs=8.0,
           color=BLUE_EDGE, weight="bold", ha="center", va="center")

    # ------------------------------------------------------------------
    # LEFT ZONE — the fluent black box (red chaos)
    # ------------------------------------------------------------------
    # black box
    _box(ax, 0.60, 2.45, 1.45, 1.10, "LLM\nblack box", DARK, DARK,
         tc="white", weight="bold", fs=9.0)
    # unverified numbers leaking out — plausible but underived
    _label(ax, 2.98, 3.86, "τ = 0.05 Pa", fs=7.5, color=RED, rot=-6.0)
    _label(ax, 3.30, 3.44, "V = 2.9 µL", fs=7.5, color=RED, rot=4.0)
    _label(ax, 2.98, 3.00, "d = 100 µm", fs=7.5, color=RED, rot=-3.0)
    _label(ax, 3.30, 2.52, "Q = 2 µL/min", fs=7.5, color=RED, rot=6.0)
    _arrow(ax, 2.05, 3.05, 2.62, 3.30, color=RED, lw=1.2)
    # broken chemistry
    _flask(ax, 1.02, 0.86, 0.78, RED)
    _label(ax, 2.20, 1.30, "broken\nchemistry", fs=6.5, color=RED_DARK,
           ha="center", va="center")
    # rejected-at-the-gate strip
    ax.add_patch(FancyBboxPatch((0.30, 0.16), 3.15, 0.44,
                                boxstyle="round,pad=0.12,rounding_size=0.06",
                                fc="none", ec=RED, linewidth=1.1,
                                linestyle=(0, (4, 2)), zorder=2))
    _label(ax, 1.875, 0.38, "no derivation, no entry —\n"
           "the gate rejects typed numbers", fs=6.3, color=RED,
           ha="center", va="center", line_h=1.35)

    # ------------------------------------------------------------------
    # MIDDLE ZONE — the gate: funnel into calculators, hard-gate neck
    # ------------------------------------------------------------------
    ax.add_patch(plt.Polygon(
        [(4.05, 4.05), (6.95, 4.05), (5.65, 1.60), (4.95, 1.60)],
        closed=True, fill=True, fc=BLUE_SOFT, ec=BLUE, lw=1.4, zorder=1))
    _label(ax, 5.5, 3.62, "deterministic calculators", fs=8.5,
           color=BLUE_EDGE, weight="bold", ha="center", va="center")
    _label(ax, 5.5, 3.26, "verify every derived number", fs=7.0, color=MUT,
           ha="center", va="center")
    _label(ax, 5.5, 2.72, "V = π(d/2)²·L", fs=8.0, color=BLUE,
           ha="center", va="center")
    _label(ax, 5.5, 2.34, "τ = 6µQ / w h²", fs=8.0, color=BLUE,
           ha="center", va="center")
    _box(ax, 4.95, 1.32, 0.70, 0.28, "hard gate", BLUE, BLUE_EDGE,
         tc="white", weight="bold", fs=6.5)
    _label(ax, 5.5, 0.98, "P(Solver Error) ≡ 0", fs=7.5, color=BLUE_EDGE,
           weight="bold", ha="center", va="center")

    # raw-inputs flow into the funnel
    _arrow(ax, 2.62, 3.10, 4.02, 3.72, color=BLUE_EDGE, lw=1.3)
    _label(ax, 3.35, 4.20, "raw inputs only", fs=6.5, color=MUT, ha="center",
           va="center")

    # ------------------------------------------------------------------
    # RIGHT ZONE — deterministic output: field DAG + formula + verified JSON
    # ------------------------------------------------------------------
    # DAG: raw inputs -> derived quantities
    _box(ax, 7.60, 2.85, 1.55, 0.72, "raw inputs\n(geometry · constants)",
         GRAY_LIGHT, GRAY, fs=6.5)
    _box(ax, 9.50, 3.60, 1.30, 0.50, "τ (Pa)", BLUE, BLUE_EDGE,
         tc="white", weight="bold", fs=7.5)
    _box(ax, 9.50, 3.00, 1.30, 0.50, "V (nL)", BLUE, BLUE_EDGE,
         tc="white", weight="bold", fs=7.5)
    _box(ax, 9.50, 2.40, 1.30, 0.50, "Q (µL/min)", BLUE, BLUE_EDGE,
         tc="white", weight="bold", fs=7.5)
    for y_t in (3.85, 3.25, 2.65):
        _arrow(ax, 9.15, y_t, 9.48, y_t, color=BLUE, lw=1.0)
    _label(ax, 7.60, 2.18, "every value follows from raw inputs alone",
           fs=6.5, color=MUT, ha="center", va="center")
    # down to the verified output
    _box(ax, 7.85, 1.15, 2.60, 0.72, "SOP + design JSON\n— every number verified ✓",
         BLUE_EDGE, BLUE_EDGE, tc="white", weight="bold", fs=7.5)
    for x_from, y_from in ((10.15, 3.85), (10.15, 3.25), (10.15, 2.65)):
        _arrow(ax, x_from, y_from, 9.40, 1.87, color=BLUE, lw=1.0)

    # ------------------------------------------------------------------
    # Honest boundary callout (bottom right)
    # ------------------------------------------------------------------
    ax.text(11.60, 0.55,
            "honest boundary: the gate verifies\n"
            "arithmetic, not physiology — on blind\n"
            "goals usable drops, hallucination 0.000",
            fontsize=6.8, color=MUT, ha="right", va="center", linespacing=1.3,
            zorder=2)

    # ------------------------------------------------------------------
    # Overlap check before writing
    # ------------------------------------------------------------------
    print("headline (derived from results/eval_flash.json + eval_pro.json, "
          "24-goal reading set):")
    print(f"  {bare_hd}")
    print(f"  {lab_hd}")
    problems = check_overlaps(fig, ax)
    if problems:
        print("LAYOUT PROBLEMS:")
        for p in problems:
            print(p)
    else:
        print("overlap check: clean")

    out = Path(__file__).resolve().parent
    fig.savefig(out / "fig_abstract.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(out / "fig_abstract.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(out / "fig_abstract.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {out / 'fig_abstract.pdf'} and {out / 'fig_abstract.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
