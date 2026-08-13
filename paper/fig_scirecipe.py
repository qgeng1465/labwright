"""Render the SciRecipe reverse-verification figure.

Turns ``results/eval_scirecipe_audit.json`` into a two-panel figure. Left: the
funnel — from all protocol summaries, how many carry numbers, how many route to
a domain we can recompute, how many were audited, how many *stated a derived
number* (the only rows a verdict can speak to), and how many of those were
*checkable*. Right: the verdict distribution over audited rows (ok /
review_required / unverifiable) with the counts set in white inside the
segments. "ok" counts only rows that asserted a derived number which re-computed
within tolerance — rows that stated no derived number are unverifiable, never
vacuously "ok" (625 of the first run's 655 "ok" rows had no claims; the honest
consistency is 30/104 = 0.288, not 0.898). The top contradictions are quoted
verbatim in the figure's bottom margin — outside both panels so they can never
collide with the bars — and an arrow draws the eye from that note up to the
unverifiable (gray) segment they live in.

Layout invariants (kept in the code so the figure can't drift):
- ``gridspec_kw={'wspace': 0.4, 'width_ratios': [1, 2]}`` on ``figsize=(16, 6)``:
  the verdict panel is twice as wide as the funnel, and the gap keeps the funnel
  labels clear of the right panel;
- the two panels share one axes row, so their bar baselines (and heights) align;
- the contradiction note is a global ``fig.text`` in the bottom margin (y≈0.05),
  below ``bottom=0.24``, with a figure-space arrow pointing up to the gray bar —
  never inside a panel where it would overlap the verdict bar.

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

FIELD_NAMES = {
    "shear_pa": "shear", "reynolds": "Re", "pressure_drop_pa": "ΔP",
    "residence_time_s": "t_res", "channel_volume_ul": "V_ch",
    "mean_velocity_mms": "ū", "seed_per_well": "seed count",
    "medium_volume_per_well_ml": "medium volume",
    "expected_confluence_pct": "confluence",
}


def _ellipsis(s: str, n: int = 74) -> str:
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


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
    # Rows that asserted at least one *derived* number (the only rows a verdict
    # can speak to); the funnel stages this so "ok" is never counted on empty
    # claims. n_ok now means "stated derived numbers that all re-computed".
    n_claimed = d.get("n_stated_derived", sum(1 for r in d.get("rows", []) if r.get("has_claims")))

    # No constrained_layout: it silently ignores the subplots_adjust below (which
    # would leave the legend overlapping the panels). Manual fractions instead.
    # Right panel is twice as wide as the funnel; the wspace keeps the long
    # funnel labels clear of the verdict bar.
    fig, axes = plt.subplots(
        1, 2, figsize=(16, 6),
        gridspec_kw={"wspace": 0.4, "width_ratios": [1, 2]},
    )
    fig.patch.set_facecolor("white")
    # top margin for the legend+title, bottom margin for the global contradiction
    # note and its arrow (the note sits below bottom=0.24, never in a panel).
    fig.subplots_adjust(top=0.82, bottom=0.24, left=0.05, right=0.98)

    # ---- left: funnel (each stage's % states its denominator) ----
    ax = axes[0]
    stages = [
        ("all protocols", total, None),
        ("numeric", numeric, f"{100.0 * numeric / total:.0f}% of all"),
        ("audited (culture + flow)", dom, f"{100.0 * dom / numeric:.0f}% of numeric"),
        ("stated a derived number", n_claimed, f"{100.0 * n_claimed / audited:.0f}% of audited"),
        ("checkable", n_check, f"{100.0 * n_check / n_claimed:.0f}% of stated"),
    ]
    ymax = stages[0][1]
    for i, (name, value, pct) in enumerate(stages):
        w = value / ymax
        c = ax.barh(i, w, height=0.6, color=BLUE, ec="none")
        outside = name + (f" · {pct}" if pct else "")
        if w >= 0.06:
            # wide enough for the count in white inside the bar
            ax.bar_label(c, labels=[f"{value:,}"], label_type="center",
                         color="white", fontsize=12, fontweight="bold")
            ax.text(w + 0.01, i, outside, va="center", ha="left", fontsize=10,
                    color=INK)
        else:
            # thin bar: keep the count in the outside label instead
            ax.text(w + 0.01, i, f"{value:,} · {outside}", va="center", ha="left",
                    fontsize=10, color=INK)
    ax.set_ylim(-0.6, len(stages) - 0.4)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("protocol audit funnel", fontsize=12, color=INK, pad=20)

    # ---- right: verdicts over the audited rows (the funnel's audited stage) ----
    # Counts are set via bar_label (white, centered, bold); a slice only carries
    # its count inside when it is wide enough — the thin needs-review slice
    # floats its count above the bar instead.
    ax = axes[1]
    counts = [(n_ok, GOOD, "verified"), (n_review, WARN, "needs review"),
              (n_unv, NEUTRAL, "unverifiable")]
    xmax = audited * 1.12  # the axis xlim, in the same data units as value
    bottom = 0
    thin: list[tuple[int, str]] = []  # slices too narrow to label in place
    for value, color, label in counts:
        if not value:
            continue
        c = ax.barh([0], [value], left=[bottom], color=color, height=0.72,
                    ec="white", lw=1.2)
        frac = value / xmax
        cx = bottom + value / 2
        if frac >= 0.06:
            # wide enough for the count inside the slice
            ax.bar_label(c, labels=[f"{value:,}"], label_type="center",
                         color="white", fontsize=12, fontweight="bold")
            if frac >= 0.09:
                # and for the slice name, under the count
                ax.text(cx, -0.22, label, ha="center", va="center", fontsize=9.5,
                        color="white" if color in (GOOD, WARN) else INK)
        else:
            thin.append((value, label))
        bottom += value
    # Thin slices (ok + review are 30 and 74 of 5,700 here): their centres sit
    # ~52 data units apart, so two floating labels at cx collide. Render ONE
    # combined annotation anchored at the bar's left edge above it instead —
    # the legend and the green/amber slivers carry the colour mapping.
    if len(thin) == 1:
        (value, label), = thin
        ax.text(value / 2, 0.52, f"{value:,}", ha="center", va="center",
                fontsize=10, color=WARN, fontweight="bold")
        ax.text(value / 2, 0.28, label, ha="center", va="bottom", fontsize=9,
                color=INK)
    elif len(thin) > 1:
        ax.text(0, 0.60, "  ·  ".join(f"{v:,} {lab}" for v, lab in thin),
                ha="left", va="center", fontsize=10, color=INK,
                fontweight="bold")
    ax.set_ylim(-1.6, 0.72)
    ax.set_xlim(0, xmax)
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"verdicts over the {audited:,} audited",
                 fontsize=12, color=INK, pad=20)

    # legend (one row, above both panels)
    handles = [
        Patch(facecolor=GOOD, label="verified (numbers follow)"),
        Patch(facecolor=WARN, label="needs review"),
        Patch(facecolor=NEUTRAL, label="unverifiable"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.96),
               ncol=3, frameon=False, fontsize=10)

    # contradiction quotes: global note in the figure's bottom margin, NOT inside
    # the right panel, so it can never overlap the verdict bar.
    rows = d.get("rows", [])
    disc = [r for r in rows if r.get("verdict") == "review_required"]
    disc.sort(key=lambda r: len(r.get("discrepancy_fields", [])), reverse=True)
    quote_lines = []
    for r in disc[:3]:
        names = ", ".join(FIELD_NAMES.get(f, f) for f in r.get("discrepancy_fields", []))
        quote_lines.append(f"{names}: {_ellipsis(r.get('quote') or '')}")
    note_txt = "contradictions, verbatim:\n" + "\n".join(quote_lines) if quote_lines \
        else "no contradictions found in the audited rows"
    note = fig.text(0.5, 0.05, note_txt, fontsize=10, color=WARN, ha="center",
                    va="bottom", linespacing=1.5, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.4", fc="#faf8f5", ec=GRID,
                              lw=0.8))

    # arrow from the note up to the unverifiable (gray) segment of the right bar.
    # Both endpoints are computed in figure space after a draw, so they track the
    # actual rendered geometry instead of hand-tuned fractions.
    fig.canvas.draw()
    if n_unv > 0:
        rnd = fig.canvas.get_renderer()
        bb = note.get_window_extent(rnd)
        src = fig.transFigure.inverted().transform((bb.x0 + bb.width / 2, bb.y1 + 12))
        gray_cx = audited - n_unv / 2
        tgt = fig.transFigure.inverted().transform(
            ax.transData.transform((gray_cx, -0.36)))
        fig.add_artist(FancyArrowPatch(src, tgt, transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=14,
                                       color=MUT, lw=1.2, zorder=4))

    out = Path(__file__).resolve().parent
    fig.savefig(out / "fig_scirecipe.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(out / "fig_scirecipe.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {out / 'fig_scirecipe.pdf'} and {out / 'fig_scirecipe.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
