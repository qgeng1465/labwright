"""Render the SciRecipe reverse-verification figure.

Turns ``results/eval_scirecipe_audit.json`` into a two-panel figure. Left: the
funnel — from all protocol summaries, how many carry numbers, how many route to
a domain we can recompute, how many were audited, how many were *checkable*.
Right: the verdict distribution over audited rows (ok / review_required /
unverifiable) plus the top contradictions quoted verbatim, so "the numbers don't
follow" is shown with the exact text, not a count.

Status colours are reserved for their meaning: verified/ok=good (green),
needs-review=warning (amber), unverifiable=neutral (gray). Text is ink
tokens only.

Usage::

    python paper/fig_scirecipe.py results/eval_scirecipe_audit.json
    # writes paper/fig_scirecipe.pdf and paper/fig_scirecipe.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _font import setup_font  # noqa: E402

setup_font()

INK = "#262522"
MUT = "#8a8782"
GRID = "#d9d7d3"
GOOD = "#3a7d44"    # status: ok / consistent
WARN = "#b26a00"    # status: review_required / discrepancy
NEUTRAL = "#a8a39d"  # status: unverifiable
BLUE = "#2E5598"    # funnel bars


def main(argv: list[str]) -> int:
    if len(argv) < 1:
        print(__doc__)
        return 1
    with open(argv[0]) as fh:
        d = json.load(fh)

    total = d["n_total"]
    numeric = d["n_numeric"]
    dom = d.get("n_culture", 0) + d.get("n_flow", 0)
    audited = d.get("n_audited", dom)
    verdicts = d.get("verdict_counts", {})
    n_ok = verdicts.get("ok", 0)
    n_review = verdicts.get("review_required", 0)
    n_unv = verdicts.get("unverifiable", 0)
    n_check = n_ok + n_review

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4),
                             constrained_layout=True, width_ratios=[1.2, 1.7])
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(top=0.80)

    # ---- left: funnel (each % states its denominator) ----
    ax = axes[0]
    stages = [
        (f"all protocols  {total:,}", total),
        (f"numeric  {numeric:,}  · {100.0 * numeric / total:.0f}% of all", numeric),
        (f"audited (culture + flow)  {dom:,}  · {100.0 * dom / numeric:.0f}% of numeric", dom),
        (f"checkable  {n_check:,}  · {100.0 * n_check / audited:.0f}% of audited", n_check),
    ]
    ymax = stages[0][1]
    for i, (label, value) in enumerate(stages):
        w = value / ymax
        ax.barh(i, w, height=0.62, color=BLUE, ec="none")
        ax.text(w + 0.01, i, label, va="center", ha="left", fontsize=8.5, color=INK)
    for i in range(len(stages) - 1):
        y0 = i - 0.31
        y1 = i + 1 + 0.31
        ax.add_patch(FancyArrowPatch((0.55, y0), (0.55, y1),
                                     arrowstyle="->", mutation_scale=8,
                                     color=GRID, lw=0.8, zorder=0))
    ax.set_ylim(-0.6, len(stages) - 0.4)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("protocol audit funnel", fontsize=9.5, color=INK, pad=4)

    # ---- right: verdicts over the audited rows (the funnel's audited stage) ----
    # A slice carries its count inside only when it is wide enough to hold text;
    # a thin slice (here the 74 review-required rows, ~1% of the bar) cannot, so
    # its count floats above the bar. Coloured slices wide enough hold their name
    # under the count; light slices put the name above the bar instead; the thin
    # slice's name lives in the legend.
    ax = axes[1]
    counts = [(n_ok, GOOD, "verified"), (n_review, WARN, "needs review"),
              (n_unv, NEUTRAL, "unverifiable")]
    xmax = audited * 1.15  # the axis xlim, in the same data units as value
    bottom = 0
    for value, color, label in counts:
        ax.barh([0], [value], left=[bottom], color=color, height=0.7, ec="white", lw=1.0)
        if value:
            frac = value / xmax
            cx = bottom + value / 2
            if frac >= 0.045:
                # wide enough for the count inside the slice
                ax.text(cx, 0, f"{value:,}", va="center", ha="center", fontsize=9,
                        color="white" if color in (GOOD, WARN) else INK, zorder=3)
            else:
                # thin slice: count floats above the bar, clear of the neighbours
                ax.text(cx, 0.58, f"{value:,}", va="center", ha="center",
                        fontsize=7.5, color=WARN, zorder=3)
            if frac >= 0.055 and color in (GOOD, WARN):
                # coloured slice wide enough for the name, under the count
                ax.text(cx, -0.15, label, ha="center", va="center", fontsize=7.5,
                        color="white", zorder=3)
            elif frac >= 0.045:
                # light/wide slice: name above the bar
                ax.text(cx, 0.47, label, ha="center", va="bottom", fontsize=8,
                        color=INK)
        bottom += value
    ax.set_ylim(-1.5, 0.72)
    ax.set_xlim(0, audited * 1.15)
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"verdicts over the {audited:,} audited", fontsize=9.5, color=INK, pad=4)

    # contradiction quotes under the verdict bar, in a bounded block with prose
    # field names and an ellipsis where the text was cut.
    rows = d.get("rows", [])
    disc = [r for r in rows if r.get("verdict") == "review_required"]
    disc.sort(key=lambda r: len(r.get("discrepancy_fields", [])), reverse=True)
    FIELD_NAMES = {
        "shear_pa": "shear", "reynolds": "Re", "pressure_drop_pa": "ΔP",
        "residence_time_s": "t_res", "channel_volume_ul": "V_ch",
        "mean_velocity_mms": "ū", "seed_per_well": "seed count",
        "medium_volume_per_well_ml": "medium volume",
        "expected_confluence_pct": "confluence",
    }

    def _ellipsis(s: str, n: int = 74) -> str:
        return s if len(s) <= n else s[: n - 1].rstrip() + "…"

    quote_lines = []
    for r in disc[:3]:
        names = ", ".join(FIELD_NAMES.get(f, f) for f in r.get("discrepancy_fields", []))
        q = _ellipsis(r.get("quote") or "")
        quote_lines.append(f"{names}: {q}")
    if quote_lines:
        ax.text(0, -0.55, "contradictions, verbatim:\n" + "\n".join(quote_lines),
                fontsize=8, color=WARN, va="top", ha="left", linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.35", fc="#faf8f5", ec=GRID, lw=0.8))

    # legend
    handles = [
        Patch(facecolor=GOOD, label="verified (numbers follow)"),
        Patch(facecolor=WARN, label="needs review"),
        Patch(facecolor=NEUTRAL, label="unverifiable"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.90),
               ncol=3, frameon=False, fontsize=8.5)

    out = Path(__file__).resolve().parent
    fig.savefig(out / "fig_scirecipe.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(out / "fig_scirecipe.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {out / 'fig_scirecipe.pdf'} and {out / 'fig_scirecipe.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
