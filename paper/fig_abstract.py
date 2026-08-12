"""Render the GitHub graphical abstract for Labwright.

A single wide panel that tells the architecture story at a glance:

     goal (natural language) -> LLM proposes RAW inputs -> deterministic
     calculators compute every derived number -> verifier RE-PROVES each one
     -> SOP + design JSON.

A numbers band across the top anchors the claim (usable designs 88-100 %,
hallucination 0.000 vs bare LLM 0-12 % / ~1.0); a bottom lane shows the naive
alternatives (type the numbers / soft self-check) rejected at the hard gate,
and a callout states the honest boundary (blind goals: usable drops, hallucination
stays 0.000).

Text is ink tokens only; status colors (the verifier's red reject) are reserved
for their meaning. Layout is authored in data coordinates; before writing, a
programmatic check asserts no text bbox overlaps another text bbox or spills
outside the canvas.

Usage::

    python paper/fig_abstract.py   # writes paper/fig_abstract.pdf/.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#262522"
MUT = "#8a8782"
GRID = "#d9d7d3"
# Labwright main path — the paper's deep academic blue, shared with the
# pipeline and benchmark figures so the system's identity is one colour
# everywhere (was orange before the palette was unified).
BLUE = "#2E5598"
BLUE_EDGE = "#1f3f70"
GRAY = "#a8a39d"
GRAY_LIGHT = "#e0dcd5"
GRAY_DARK = "#74706a"
RED = "#b3261e"  # status: hard reject — reserved for its meaning
WH = "round,pad=0.25,rounding_size=0.10"

X0, X1, Y0, Y1 = 0.0, 12.0, 0.0, 5.0


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
           weight="normal", zorder=2):
    ax.text(x, y, text, fontsize=fs, color=color, ha=ha, va=va, zorder=zorder,
            fontweight=weight, linespacing=1.3)


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
    fig, ax = plt.subplots(figsize=(12.0, 5.0))
    ax.set_xlim(X0, X1)
    ax.set_ylim(Y0, Y1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ------------------------------------------------------------------
    # Top band: title + benchmark numbers
    # ------------------------------------------------------------------
    ax.add_patch(FancyBboxPatch((0.15, 4.30), 11.7, 0.58,
                                boxstyle="round,pad=0.15,rounding_size=0.08",
                                fc="#f6f4f0", ec=GRID, linewidth=1.0, zorder=1))
    _label(ax, 0.45, 4.59, "one yardstick, every row:", fs=8.0, color=MUT,
           va="center")
    _label(ax, 2.55, 4.59, "bare LLM  0–12 % usable  ·  hallucination ~1.0",
           fs=9.0, color=INK, va="center")
    _label(ax, 11.60, 4.59,
           "Labwright  88–100 % usable  ·  hallucination 0.000", fs=10.0,
           color=BLUE_EDGE, va="center", weight="bold", ha="right")

    # ------------------------------------------------------------------
    # Main pipeline row
    # ------------------------------------------------------------------
    row_y = 2.35
    row_h = 1.05

    _box(ax, 0.30, row_y, 1.35, row_h, "goal\n(natural\nlanguage)", GRAY_LIGHT,
         GRAY, fs=8.5)
    _box(ax, 2.05, row_y, 1.75, row_h, "LLM agent\nproposes\nraw inputs",
         BLUE, BLUE_EDGE, tc="white", weight="bold", fs=9.5)
    _box(ax, 4.20, row_y, 1.75, row_h, "deterministic\ncalculators\n(physics)",
         BLUE, BLUE_EDGE, tc="white", weight="bold", fs=9.5)
    _box(ax, 6.35, row_y, 2.00, row_h, "verifier\nre-proves\nevery number",
         BLUE, BLUE_EDGE, tc="white", weight="bold", fs=9.5)
    _box(ax, 9.00, row_y, 2.55, row_h, "SOP + design JSON\nverified",
         GRAY_DARK, GRAY_DARK, tc="white", weight="bold", fs=8.5)

    # flow arrows
    box_rights = [1.65, 3.80, 5.95, 8.35, 11.55]
    box_lefts = [0.30, 2.05, 4.20, 6.35, 9.00]
    for i in range(4):
        _arrow(ax, box_rights[i] + 0.06, row_y + row_h / 2,
               box_lefts[i + 1] - 0.06, row_y + row_h / 2, color=BLUE_EDGE)

    # raw / derived lane labels under the row
    _label(ax, 2.925, row_y - 0.30, "the model never writes a derived number",
           fs=7.5, color=MUT, ha="center", va="top")
    _label(ax, 8.20, row_y - 0.30,
           "every derived number is recomputed from the model's own raw inputs",
           fs=7.5, color=MUT, ha="center", va="top")

    # ------------------------------------------------------------------
    # Retry loop: rejected design returns to the agent
    # ------------------------------------------------------------------
    loop_y = 1.34
    _arrow(ax, 7.35, row_y - 0.05, 7.35, loop_y + 0.05, color=MUT, lw=1.1)
    _arrow(ax, 7.35, loop_y, 2.925, loop_y, color=MUT, lw=1.1, style="-|>")
    _arrow(ax, 2.925, loop_y, 2.925, row_y - 0.05, color=MUT, lw=1.1)
    _label(ax, 5.15, loop_y + 0.12, "invalid design → agent retries",
           fs=7.5, color=MUT, ha="center", va="bottom")

    # ------------------------------------------------------------------
    # Naive alternatives lane + reject (bottom left/middle)
    # ------------------------------------------------------------------
    _label(ax, 0.30, 0.92, "the alternatives an LLM reaches for — rejected:",
           fs=7.5, color=MUT, va="top")
    _box(ax, 0.30, 0.30, 1.35, 0.42, "type the\nnumbers", GRAY_LIGHT, GRAY, fs=7.5)
    _box(ax, 2.05, 0.30, 1.75, 0.42, "soft self-check\n(LLM judges its own work)",
         GRAY_LIGHT, GRAY, fs=7.5)
    ax.add_patch(FancyBboxPatch((4.45, 0.18), 3.40, 0.62,
                                boxstyle="round,pad=0.15,rounding_size=0.08",
                                fc="none", ec=RED, linewidth=1.4,
                                linestyle=(0, (4, 2)), zorder=2))
    _label(ax, 6.15, 0.63, "REJECTED", fs=9.0, color=RED, ha="center", va="center",
           weight="bold")
    _label(ax, 6.15, 0.36, "a number you typed is not trusted", fs=7.5, color=RED,
           ha="center", va="center")

    # arrows from alternatives toward the reject
    _arrow(ax, 1.71, 0.51, 2.05 - 0.06, 0.51, color=MUT, lw=1.0)
    _arrow(ax, 3.86, 0.51, 4.45 - 0.06, 0.51, color=RED, lw=1.0)

    # ------------------------------------------------------------------
    # Honest boundary callout (bottom right, its own region)
    # ------------------------------------------------------------------
    ax.text(11.70, 0.52,
            "honest boundary: on blind goals the gate can't\nsupply physiology "
            "the model doesn't know — usable\ndrops, hallucination stays 0.000",
            fontsize=6.8, color=MUT, ha="right", va="center", linespacing=1.3,
            zorder=2)

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
    fig.savefig(out / "fig_abstract.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(out / "fig_abstract.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {out / 'fig_abstract.pdf'} and {out / 'fig_abstract.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
