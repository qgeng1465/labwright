"""Render the paper's pipeline figure (Fig 1): propose -> calculate -> verify.

A horizontal flow diagram. The Labwright path (proposal -> deterministic
calculators -> hard verifier) is drawn as connected boxes in the saturated
Labwright orange; the naive alternatives (bare / soft-gate / self-verify) are a
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

INK = "#262522"
MUT = "#8a8782"
GRID = "#d9d7d3"
ORANGE = "#eb6834"
ORANGE_EDGE = "#c85a22"
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
    fig, ax = plt.subplots(figsize=(9.2, 3.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ---- goal entry (left) ----
    _box(ax, 0.15, 1.35, 0.95, 0.5, "goal\n(natural language)", GRAY_LIGHT, GRAY, fs=8.5)

    # ---- LLM agent proposes raw inputs ----
    _box(ax, 1.75, 1.35, 1.15, 0.5, "LLM agent\nproposes raw inputs", ORANGE, ORANGE_EDGE, tc="white", weight="bold", fs=9)

    # ---- calculators ----
    _box(ax, 3.55, 1.35, 1.15, 0.5, "deterministic\ncalculators", ORANGE, ORANGE_EDGE, tc="white", weight="bold", fs=9)

    # ---- verifier (hard gate) ----
    _box(ax, 5.35, 1.35, 1.35, 0.5, "verifier\nre-derives every number", ORANGE, ORANGE_EDGE, tc="white", weight="bold", fs=9)

    # ---- accepted output (right) ----
    _box(ax, 7.5, 1.35, 1.6, 0.5, "SOP + design JSON\n(verified)", GRAY_DARK, GRAY_DARK, tc="white", weight="bold", fs=8.5)

    # main flow arrows
    _arrow(ax, 1.10, 1.60, 1.75, 1.60, color=ORANGE_EDGE)
    _arrow(ax, 2.90, 1.60, 3.55, 1.60, color=ORANGE_EDGE)
    _arrow(ax, 4.70, 1.60, 5.35, 1.60, color=ORANGE_EDGE)
    _arrow(ax, 6.70, 1.60, 7.50, 1.60, color=ORANGE_EDGE)

    # ---- the naive alternatives lane (gray) ----
    ax.text(0.15, 0.62, "naive alternatives\n(bare / soft-gate / self-verify):",
            fontsize=7.5, color=MUT, va="top", ha="left")
    _box(ax, 3.55, 0.55, 1.15, 0.35, "type the\nnumbers", GRAY_LIGHT, GRAY, fs=7.5)
    _box(ax, 5.35, 0.55, 1.35, 0.35, "no external\nverifier", GRAY_LIGHT, GRAY, fs=7.5)
    ax.add_patch(FancyBboxPatch(
        (7.05, 0.35), 2.0, 0.55, boxstyle="round,pad=0.2,rounding_size=0.1",
        fc="none", ec=RED, linewidth=1.4, linestyle=(0, (4, 2)), zorder=2,
    ))
    ax.text(8.05, 0.625, "REJECTED", ha="center", va="center",
            fontsize=8.5, color=RED, zorder=4, fontweight="bold")
    ax.text(8.05, 0.30, "numbers you type are not trusted",
            ha="center", va="top", fontsize=7, color=RED, zorder=4)

    # retry loop on rejected input
    _arrow(ax, 5.35, 1.30, 5.35, 0.95, color=MUT, style="-|>", lw=1.0)
    _arrow(ax, 5.35, 0.95, 2.35, 0.95, color=MUT, style="-|>", lw=1.0)
    ax.text(3.6, 0.98, "invalid submission -> agent retries",
            fontsize=7, color=MUT, ha="center", va="bottom")

    out = Path(__file__).resolve().parent
    fig.savefig(out / "fig_pipeline.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(out / "fig_pipeline.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {out / 'fig_pipeline.pdf'} and {out / 'fig_pipeline.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
