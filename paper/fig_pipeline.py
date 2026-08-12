"""Render the paper's pipeline figure (Fig 1): propose -> calculate -> verify.

A horizontal flow diagram. The Labwright path (proposal -> deterministic
calculators -> hard verifier) is drawn as connected boxes in the saturated
Labwright deep blue; the naive alternatives (bare / soft-gate / self-verify) are a
gray lane underneath that ends at the verifier's reject — "numbers you type are
not trusted". Two loops are shown: an accepted design exits to SOP + design
JSON; an invalid submission loops back into the agent for retry.

Text uses ink tokens only; the one status color (the verifier's reject) is
reserved for its meaning, not used as a series color.

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
BLUE = "#2E5598"       # Labwright main path — the paper's deep academic blue
BLUE_EDGE = "#1f3f70"
GRAY = "#a8a39d"
GRAY_LIGHT = "#e0dcd5"
GRAY_DARK = "#74706a"
RED = "#b3261e"  # status: hard reject — used only for its meaning

WH = "round,pad=0.28,rounding_size=0.12"


def _box(ax, x, y, w, h, text, fc, ec, tc=INK, fs=9.5, weight="normal"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=WH, fc=fc, ec=ec, linewidth=1.2, zorder=3,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=4, fontweight=weight, linespacing=1.3)


def _arrow(ax, x1, y1, x2, y2, color=MUT, style="-|>", lw=1.4, zorder=2, ls="-"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13,
        color=color, linewidth=lw, zorder=zorder, linestyle=ls,
    ))


def main() -> int:
    # A taller canvas than the original so the retry loop has a dedicated lane
    # between the main row and the naive-alternatives row — the retry label no
    # longer sits on top of its own arrow. xlim/ylim start slightly negative so
    # the rounded-patch padding on the goal box and the retry lane are not
    # clipped at the axes edge (bbox_inches="tight" still trims to content).
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 3.6)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ---- goal entry (left) ----
    _box(ax, 0.15, 1.75, 0.95, 0.5, "goal\n(natural language)", GRAY_LIGHT, GRAY, fs=9)

    # ---- LLM agent proposes raw inputs ----
    _box(ax, 1.75, 1.75, 1.15, 0.5, "LLM agent\nproposes raw inputs", BLUE, BLUE_EDGE, tc="white", weight="bold", fs=9)

    # ---- calculators ----
    _box(ax, 3.55, 1.75, 1.15, 0.5, "deterministic\ncalculators", BLUE, BLUE_EDGE, tc="white", weight="bold", fs=9)

    # ---- verifier (hard gate) ----
    _box(ax, 5.35, 1.75, 1.35, 0.5, "verifier\nre-derives every number", BLUE, BLUE_EDGE, tc="white", weight="bold", fs=9)

    # ---- accepted output (right) ----
    _box(ax, 7.5, 1.75, 1.6, 0.5, "SOP + design JSON\n(verified)", GRAY_DARK, GRAY_DARK, tc="white", weight="bold", fs=9)

    # main flow arrows
    _arrow(ax, 1.10, 2.00, 1.75, 2.00, color=BLUE_EDGE)
    _arrow(ax, 2.90, 2.00, 3.55, 2.00, color=BLUE_EDGE)
    _arrow(ax, 4.70, 2.00, 5.35, 2.00, color=BLUE_EDGE)
    _arrow(ax, 6.70, 2.00, 7.50, 2.00, color=BLUE_EDGE)

    # ---- the naive alternatives lane (gray) ----
    ax.text(0.15, 0.95, "naive alternatives\n(bare / soft-gate / self-verify):",
            fontsize=8, color=MUT, va="top", ha="left")
    _box(ax, 3.55, 0.85, 1.15, 0.35, "type the\nnumbers", GRAY_LIGHT, GRAY, fs=8)
    _box(ax, 5.35, 0.85, 1.35, 0.35, "no external\nverifier", GRAY_LIGHT, GRAY, fs=8)
    ax.add_patch(FancyBboxPatch(
        (7.05, 0.65), 2.0, 0.55, boxstyle="round,pad=0.2,rounding_size=0.1",
        fc="none", ec=RED, linewidth=1.4, linestyle=(0, (4, 2)), zorder=2,
    ))
    ax.text(8.05, 0.925, "REJECTED", ha="center", va="center",
            fontsize=9, color=RED, zorder=4, fontweight="bold")
    ax.text(8.05, 0.60, "unverified numbers are rejected",
            ha="center", va="top", fontsize=8, color=RED, zorder=4)
    # connect the naive lane: type the numbers -> no external verifier ->
    # REJECTED. The boxes carry rounded-patch padding, so arrows run between the
    # padded box edges (4.98 / 5.07 for the first pair, 6.98 / 6.85 for the
    # second) and terminate on the REJECTED box's left edge.
    _arrow(ax, 4.98, 1.02, 5.07, 1.02, color=MUT, style="-|>", lw=1.0)
    _arrow(ax, 6.98, 1.02, 6.85, 0.925, color=MUT, style="-|>", lw=1.0)

    # retry loop: own lane at y=1.42 between the main row (boxes bottom 1.75)
    # and the naive row (boxes top 1.48), so the label never collides. The
    # horizontal leg runs plain and the arrowhead is carried up the vertical
    # leg, terminating inside the LLM box's padded fill — it reads as re-entering
    # the agent, not floating below it.
    _arrow(ax, 5.35, 1.70, 5.35, 1.42, color=MUT, style="-", lw=1.0)
    _arrow(ax, 5.35, 1.42, 2.35, 1.42, color=MUT, style="-", lw=1.0)
    # zorder 5: the head must land inside the LLM box's padded fill and stay
    # visible on top of it, not hide behind the box (zorder 3).
    _arrow(ax, 2.35, 1.42, 2.35, 1.72, color=MUT, style="-|>", lw=1.0, zorder=5)
    ax.text(3.85, 1.52, "invalid submission → agent retries",
            fontsize=8, color=MUT, ha="center", va="bottom")

    out = Path(__file__).resolve().parent
    fig.savefig(out / "fig_pipeline.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(out / "fig_pipeline.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {out / 'fig_pipeline.pdf'} and {out / 'fig_pipeline.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
