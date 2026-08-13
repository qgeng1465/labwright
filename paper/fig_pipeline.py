"""Render the paper's pipeline figure (Fig 1): propose -> calculate -> verify.

A two-tier flow diagram. The Labwright path (goal -> proposal -> deterministic
calculators -> hard verifier -> SOP + design JSON) is drawn as connected boxes in
the saturated Labwright deep blue across the upper half; the naive alternatives
(bare / soft-gate / self-verify) are a gray lane in the lower half that ends at
the verifier's hard reject — "numbers you type are not trusted". A light-gray
dashed rule physically separates the two tiers. Two loops are shown: an accepted
design exits to SOP + design JSON on the right; an invalid submission loops back
from the verifier to the agent for retry, staying inside the upper tier so it
never threads through the gray lane.

Layout invariants (kept in the code so the figure can't drift):
- the left edges of the goal entry, the naive lane and its first box share x0;
- the right edges of the SOP output and the REJECTED box share x1;
- the retry loop owns a lane below the main row and above the separator, so it
  never crosses the gray boxes below.

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
BLUE_LIGHT = "#c3d2ec"  # the goal entry: part of the blue tier, but an input
GRAY = "#a8a39d"
GRAY_LIGHT = "#e0dcd5"
RED = "#b3261e"  # status: hard reject — used only for its meaning


def _box(ax, x0, y0, x1, y1, text, fc, ec, tc=INK, fs=9.5, weight="normal",
         pad=0.28):
    """Rounded box whose *visible* extent is exactly [x0,x1]x[y0,y1].

    FancyBboxPatch pads outward from the box you give it, so it is fed a box
    shrunk by ``pad``; every arrow in this figure connects the visible edges
    returned here, keeping alignment (left/right) exact on the canvas.
    """
    ax.add_patch(FancyBboxPatch(
        (x0 + pad, y0 + pad), (x1 - x0) - 2 * pad, (y1 - y0) - 2 * pad,
        boxstyle=f"round,pad={pad},rounding_size=0.12",
        fc=fc, ec=ec, linewidth=1.2, zorder=3,
    ))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=4, fontweight=weight, linespacing=1.3)


def _arrow(ax, x1, y1, x2, y2, color=MUT, style="-|>", lw=1.4, zorder=2, ls="-"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13,
        color=color, linewidth=lw, zorder=zorder, linestyle=ls,
    ))


def main() -> int:
    # Two tiers: blue main flow on top, gray alternatives below, separated by a
    # dashed rule. All box coords below are *visible* edges (see _box).
    fig, ax = plt.subplots(figsize=(9.4, 4.3))
    ax.set_xlim(-0.7, 10.9)
    ax.set_ylim(-0.55, 4.05)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ---- upper tier: the Labwright main flow (blue) ----
    Y_TOP, Y_BOT = 3.35, 2.75          # main-row visible band
    Y_MID = (Y_TOP + Y_BOT) / 2
    _box(ax, 0.25, Y_BOT, 1.30, Y_TOP, "goal\n(natural language)",
         BLUE_LIGHT, BLUE_EDGE, fs=9)
    _box(ax, 2.10, Y_BOT, 3.30, Y_TOP, "LLM agent\nproposes raw inputs",
         BLUE, BLUE_EDGE, tc="white", weight="bold", fs=9)
    _box(ax, 4.00, Y_BOT, 5.20, Y_TOP, "deterministic\ncalculators",
         BLUE, BLUE_EDGE, tc="white", weight="bold", fs=9)
    _box(ax, 5.90, Y_BOT, 7.40, Y_TOP, "verifier\nre-derives every number",
         BLUE, BLUE_EDGE, tc="white", weight="bold", fs=9)
    _box(ax, 8.45, Y_BOT, 10.25, Y_TOP, "SOP + design JSON\n(verified)",
         BLUE, BLUE_EDGE, tc="white", weight="bold", fs=9)

    # main-flow arrows
    for xa, xb in [(1.30, 2.10), (3.30, 4.00), (5.20, 5.90), (7.40, 8.45)]:
        _arrow(ax, xa, Y_MID, xb, Y_MID, color=BLUE_EDGE)

    # ---- retry loop: own lane in the upper tier (never crosses the gray lane) ----
    LANE = 2.35
    _arrow(ax, 6.65, Y_BOT, 6.65, LANE, color=MUT, style="-", lw=1.0)
    _arrow(ax, 6.65, LANE, 2.70, LANE, color=MUT, style="-", lw=1.0)
    _arrow(ax, 2.70, LANE, 2.70, Y_BOT, color=MUT, style="-|>", lw=1.0, zorder=5)
    ax.text(4.55, LANE + 0.07, "invalid submission → agent retries",
            fontsize=7.5, color=MUT, ha="center", va="bottom")

    # ---- dashed rule physically separating the two tiers ----
    ax.plot([-0.6, 10.85], [2.0, 2.0], color=GRID, lw=0.8, ls=(0, (4, 3)),
            zorder=1)

    # ---- lower tier: naive alternatives (gray) -> REJECTED (red) ----
    ax.text(0.25, 1.58, "naive alternatives\n(bare / soft-gate / self-verify):",
            fontsize=8, color=MUT, ha="left", va="bottom")
    _box(ax, 0.25, 0.75, 1.60, 1.15, "type the\nnumbers", GRAY_LIGHT, GRAY,
         fs=8, pad=0.22)
    _box(ax, 2.30, 0.75, 3.90, 1.15, "no external\nverifier", GRAY_LIGHT, GRAY,
         fs=8, pad=0.22)
    _arrow(ax, 1.60, 0.95, 2.30, 0.95, color=MUT, style="-|>", lw=1.0)
    _arrow(ax, 3.90, 0.95, 8.00, 0.95, color=MUT, style="-|>", lw=1.0)

    # REJECTED: dashed red box in the lower-right tier, clear of the main flow
    # above and of its own caption below.
    pad = 0.2
    ax.add_patch(FancyBboxPatch(
        (8.00 + pad, 0.62 + pad), (10.25 - 8.00) - 2 * pad, (1.32 - 0.62) - 2 * pad,
        boxstyle=f"round,pad={pad},rounding_size=0.12", fc="none", ec=RED,
        linewidth=1.4, linestyle=(0, (4, 2)), zorder=3,
    ))
    ax.text(9.125, 0.97, "REJECTED", ha="center", va="center",
            fontsize=9.5, color=RED, zorder=4, fontweight="bold")
    ax.text(9.125, 0.44, "unverified numbers are rejected",
            ha="center", va="top", fontsize=8, color=RED, zorder=4)

    out = Path(__file__).resolve().parent
    fig.savefig(out / "fig_pipeline.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(out / "fig_pipeline.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {out / 'fig_pipeline.pdf'} and {out / 'fig_pipeline.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
