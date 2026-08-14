"""Render the SciRecipe reverse-verification figure.

Turns ``results/eval_scirecipe_audit.json`` into a two-panel figure. Left: the
funnel — from all protocol summaries, how many carry numbers, how many route to
a domain we can recompute, how many were audited, how many *stated a derived
number* (the only rows a verdict can speak to), and how many of those were
*checkable*. Right: what the gate found. The old single stacked bar collapsed
to one 98% gray slab with two invisible slivers and a floating annotation; the
new version splits it into two aligned rows with an explicit zoom connector:

* row 1 — ``all audited (5,700)``: unverifiable 5,596 (98.2%) + a thin
  checkable sliver (104, 1.8%);
* row 2 — ``checkable (104)``, drawn at full width: verified 30 (29%) +
  needs review 74 (71%), so the honest 30/104 = 0.288 consistency is visible
  instead of buried at 0.5% of a bar;
* the headline under row 2 states that consistency directly.

"ok" counts only rows that asserted a derived number which re-computed within
tolerance — rows that stated no derived number are unverifiable, never
vacuously "ok" (625 of the first run's 655 "ok" rows had no claims; the honest
consistency is 30/104 = 0.288, not 0.898). Example contradiction quotes sit in
a boxed footnote in the bottom margin (outside both panels) with no arrow — the
arrow in the old version pointed at the *unverifiable* slab even though the
quotes come from the needs-review rows.

Layout invariants (kept in the code so the figure can't drift):
- ``gridspec_kw={'wspace': 0.32, 'width_ratios': [1, 1.5]}`` on
  ``figsize=(16, 6.5)``; the gap keeps the funnel labels clear of the right panel;
- every count and percentage is recomputed from the committed JSON in ``main``,
  never re-typed;
- the contradiction note is a global ``fig.text`` in the bottom margin, below
  ``bottom=0.20``, never inside a panel.

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
from matplotlib.patches import Patch

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
BLUE_LIGHT = "#c3d2ec"  # the checkable sliver within the audited row

FIELD_NAMES = {
    "shear_pa": "shear", "reynolds": "Re", "pressure_drop_pa": "ΔP",
    "residence_time_s": "t_res", "channel_volume_ul": "V_ch",
    "mean_velocity_mms": "ū", "seed_per_well": "seed count",
    "medium_volume_per_well_ml": "medium volume",
    "expected_confluence_pct": "confluence",
}


def _ellipsis(s: str, n: int = 60) -> str:
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
    vc = d.get("verdict_counts", {})
    n_ok = vc.get("ok", 0)
    n_review = vc.get("review_required", 0)
    n_unv = vc.get("unverifiable", 0)
    n_check = n_ok + n_review  # == d["n_checkable"]
    n_claimed = d.get("n_stated_derived", 0)

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(16, 6.5),
        gridspec_kw={"wspace": 0.32, "width_ratios": [1, 1.5]},
    )
    fig.patch.set_facecolor("white")
    # top margin for the legend, bottom margin for the contradiction footnote.
    fig.subplots_adjust(top=0.84, bottom=0.20, left=0.04, right=0.97)

    # ---- left: funnel (each stage's % states its denominator) ----
    stages = [
        ("all protocols", total, None),
        ("numeric", numeric, f"{100.0 * numeric / total:.0f}% of all"),
        ("audited · culture + flow", dom, f"{100.0 * dom / numeric:.0f}% of numeric"),
        ("stated a derived number", n_claimed, f"{100.0 * n_claimed / audited:.0f}% of audited"),
        ("checkable", n_check, f"{100.0 * n_check / n_claimed:.0f}% of stated"),
    ]
    ymax = stages[0][1]
    for i, (name, value, pct) in enumerate(stages):
        w = value / ymax
        ax0.barh(i, w, height=0.55, color=BLUE, ec="none", zorder=3)
        if w >= 0.055:
            ax0.text(w / 2, i, f"{value:,}", ha="center", va="center",
                     color="white", fontsize=12, fontweight="bold", zorder=4)
            label = f"{name} · {pct}" if pct else name
        else:
            # thin bar: keep the count in the outside label instead
            label = f"{value:,} · {name}" + (f" · {pct}" if pct else "")
        ax0.text(w + 0.012, i, label, ha="left", va="center", fontsize=10,
                 color=INK, zorder=4)
    ax0.set_xlim(0, 1.32)
    ax0.set_ylim(-0.55, len(stages) - 0.45)
    ax0.invert_yaxis()
    ax0.set_xticks([])
    ax0.set_yticks([])
    for spine in ax0.spines.values():
        spine.set_visible(False)
    ax0.set_title("protocol audit funnel", fontsize=12, color=INK, pad=16)

    # ---- right: verdicts over the audited rows ----
    # Two aligned rows: the audited set (98% unverifiable) and the checkable
    # subset expanded to full width so the 30/74 split is actually visible.
    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(-0.10, 2.30)
    ax1.set_xticks([])
    ax1.set_yticks([])
    for spine in ax1.spines.values():
        spine.set_visible(False)

    # row 1: all audited — unverifiable slab + a thin checkable sliver.
    r1y, r1h = 1.60, 0.5
    frac_unv = n_unv / audited      # 0.982
    frac_chk = n_check / audited    # 0.018
    ax1.barh(r1y, frac_unv, left=0.06, height=r1h, color=NEUTRAL, ec="white",
             lw=1.0, zorder=3)
    sliver_x = 0.06 + frac_unv
    ax1.barh(r1y, frac_chk, left=sliver_x, height=r1h, color=BLUE_LIGHT,
             edgecolor=BLUE, lw=0.8, zorder=4)
    ax1.text(0.06 + frac_unv / 2, r1y, f"{n_unv:,}", ha="center", va="center",
             color="white", fontsize=11, fontweight="bold", zorder=5)
    ax1.text(0.06 + frac_unv / 2, r1y - r1h / 2 - 0.07, f"{100.0 * frac_unv:.0f}%",
             ha="center", va="top", color=MUT, fontsize=8.5)
    # the checkable sliver is ~2% of the bar — label it above, not inside.
    ax1.text(sliver_x, r1y + r1h / 2 + 0.10,
             f"checkable {n_check:,} · {100.0 * frac_chk:.1f}%", ha="center",
             va="bottom", color=BLUE, fontsize=8.5, fontweight="bold")
    ax1.text(0.5, 2.02, f"all audited ({audited:,})", ha="center", va="bottom",
             fontsize=11, color=INK, fontweight="bold")

    # dashed connector from the sliver down to the expanded checkable row.
    ax1.plot([sliver_x, sliver_x], [r1y - r1h / 2, 0.90], color=MUT, lw=0.8,
             ls=(0, (2, 2)), zorder=2)

    # row 2: the checkable subset at full width — verified vs needs review.
    r2y = 0.60
    if n_check > 0:
        f_ok = n_ok / n_check      # 0.288
        f_rev = n_review / n_check  # 0.711
        ax1.barh(r2y, f_ok, left=0.06, height=r1h, color=GOOD, ec="white",
                 lw=1.0, zorder=3)
        ax1.barh(r2y, f_rev, left=0.06 + f_ok, height=r1h, color=WARN,
                 ec="white", lw=1.0, zorder=3)
        ax1.text(0.06 + f_ok / 2, r2y, f"{n_ok:,}", ha="center", va="center",
                 color="white", fontsize=11, fontweight="bold", zorder=5)
        ax1.text(0.06 + f_ok + f_rev / 2, r2y, f"{n_review:,}", ha="center",
                 va="center", color="white", fontsize=11, fontweight="bold",
                 zorder=5)
        ax1.text(0.06 + f_ok / 2, r2y - r1h / 2 - 0.05, f"{100.0 * f_ok:.0f}%",
                 ha="center", va="top", color=INK, fontsize=9, fontweight="bold")
        ax1.text(0.06 + f_ok + f_rev / 2, r2y - r1h / 2 - 0.05,
                 f"{100.0 * f_rev:.0f}%", ha="center", va="top", color=INK,
                 fontsize=9, fontweight="bold")
    ax1.text(0.5, 0.85, f"checkable ({n_check:,})", ha="center", va="top",
             fontsize=11, color=INK, fontweight="bold")

    # headline: the honest consistency of the checkable subset.
    if n_check > 0:
        ax1.text(0.5, 0.12,
                 f"of the {n_check:,} checkable protocols, {n_ok} were "
                 f"internally consistent ({100.0 * n_ok / n_check:.0f}%)",
                 ha="center", va="center", fontsize=12.5, color=INK,
                 fontweight="bold")
    ax1.set_title("verdicts over the audited protocols", fontsize=12,
                  color=INK, pad=16)

    # legend (one row, above both panels)
    handles = [
        Patch(facecolor=GOOD, label="verified (numbers follow)"),
        Patch(facecolor=WARN, label="needs review"),
        Patch(facecolor=NEUTRAL, label="unverifiable"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.97),
               ncol=3, frameon=False, fontsize=10)

    # contradiction quotes: boxed footnote in the bottom margin, NOT inside the
    # right panel, so it can never overlap the bars. They come from the
    # needs-review rows — no arrow, the amber legend entry carries that link.
    rows = d.get("rows", [])
    disc = [r for r in rows if r.get("verdict") == "review_required"]
    disc.sort(key=lambda r: len(r.get("discrepancy_fields", [])), reverse=True)
    quote_lines = []
    for r in disc[:2]:
        names = ", ".join(FIELD_NAMES.get(f, f) for f in r.get("discrepancy_fields", []))
        quote_lines.append(f"{names}: {_ellipsis(r.get('quote') or '')}")
    note_txt = "needs review, verbatim:  " + "  |  ".join(quote_lines) if quote_lines \
        else "no contradictions found in the audited rows"
    fig.text(0.5, 0.055, note_txt, fontsize=9.5, color=WARN, ha="center",
             va="center", linespacing=1.4, zorder=5,
             bbox=dict(boxstyle="round,pad=0.35", fc="#faf8f5", ec=GRID,
                       lw=0.8))

    out = Path(__file__).resolve().parent
    fig.savefig(out / "fig_scirecipe.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(out / "fig_scirecipe.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {out / 'fig_scirecipe.pdf'} and {out / 'fig_scirecipe.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
