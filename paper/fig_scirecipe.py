"""Render the SciRecipe reverse-verification figure.

Turns ``results/eval_scirecipe_audit.json`` into a two-panel figure. Left: the
funnel — from all protocol summaries, how many carry numbers, how many route to
a domain we can recompute, how many were audited, how many were *checkable*.
Right: the verdict distribution over audited rows (ok / review_required /
unverifiable) plus the top contradictions quoted verbatim, so "the numbers don't
follow" is shown with the exact text, not a count.

Status colours are reserved for their meaning: ok/consistent=good (green),
review_required=warning (amber), unverifiable=neutral (gray). Text is ink
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

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1),
                             constrained_layout=True, width_ratios=[1.2, 1.7])
    fig.patch.set_facecolor("white")

    # ---- left: funnel ----
    ax = axes[0]
    stages = [
        (f"all protocols  {total:,}", total),
        (f"numeric  {numeric:,}  ({100.0 * numeric / total:.0f}%)", numeric),
        (f"culture + flow  {dom:,}  ({100.0 * dom / numeric:.0f}%)", dom),
        (f"checkable  {n_check:,}  ({100.0 * n_check / audited:.0f}%)", n_check),
    ]
    ymax = stages[0][1]
    for i, (label, value) in enumerate(stages):
        w = value / ymax
        ax.barh(i, w, height=0.62, color=BLUE, alpha=0.35 + 0.55 * w, ec="none")
        ax.text(w + 0.008, i, label, va="center", ha="left", fontsize=8.5, color=INK)
        ax.text(-0.008, i, f"{value:,}", va="center", ha="right", fontsize=8, color=MUT)
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
    ax.set_title("funnel", fontsize=8.5, color=INK, pad=4)

    # ---- right: verdicts ----
    ax = axes[1]
    counts = [(n_ok, GOOD, "ok"), (n_review, WARN, "review_required"), (n_unv, NEUTRAL, "unverifiable")]
    bottom = 0
    for value, color, label in counts:
        ax.barh([0], [value], left=[bottom], color=color, height=0.5, ec="white", lw=0.5)
        if value:
            ax.text(bottom + value / 2, 0, f"{value:,}", va="center", ha="center",
                    fontsize=9, color="white" if color in (GOOD, WARN) else INK, zorder=3)
        bottom += value
    ax.text(bottom + max(audited * 0.01, 3), 0,
            f"of {audited:,} audited", va="center", ha="left", fontsize=8, color=MUT)
    ax.set_ylim(-0.55, 0.55)
    ax.set_xlim(0, audited * 1.18)
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("verdicts over audited rows", fontsize=8.5, color=INK, pad=4)

    # contradiction quotes under the verdict bar
    rows = d.get("rows", [])
    disc = [r for r in rows if r.get("verdict") == "review_required"]
    disc.sort(key=lambda r: len(r.get("discrepancy_fields", [])), reverse=True)
    quote_y = -0.35
    quote_lines = []
    for r in disc[:3]:
        fields = ",".join(r.get("discrepancy_fields", []))
        q = (r.get("quote") or "")[:88]
        quote_lines.append(f"  {fields}: {q}")
    if quote_lines:
        ax.text(0, quote_y, "contradictions (verbatim):\n" + "\n".join(quote_lines),
                fontsize=6.8, color=WARN, va="top", ha="left", linespacing=1.4)

    # legend
    handles = [
        Patch(facecolor=GOOD, label="ok (numbers follow)"),
        Patch(facecolor=WARN, label="review_required"),
        Patch(facecolor=NEUTRAL, label="unverifiable"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=3, frameon=False, fontsize=8)

    out = Path(__file__).resolve().parent
    fig.savefig(out / "fig_scirecipe.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(out / "fig_scirecipe.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {out / 'fig_scirecipe.pdf'} and {out / 'fig_scirecipe.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
