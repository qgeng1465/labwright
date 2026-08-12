"""Render the paper's blind-set per-goal figure (Fig 2).

Each row is one of the 12 blind goals (no target stated in the goal); each
column is a system/model family: bare-LLM and Labwright on flash and pro. The
cell colour encodes the recovery error (relative error of the reported target
parameter vs the gold value, log-scaled and clamped at 10×; the two seeding
goals are scored on cell-count recovery). A white cell with "√" means the
reported value lands within 5% of the physical target; colour deepens with the
miss; a gray cell with "·" means the system reported nothing recoverable at all
(unverifiable — scored hallucination 1.0 under the paper's convention).

The figure carries the paper's central caveat visually: on the blind set the
bare model is *self-consistent but wrong* (it emits a plausible default chip
regardless of target), while Labwright's verifier keeps hallucination at zero
but the usable rate collapses because the blind goals require domain knowledge
the calculators cannot supply. Both are honest findings; the figure makes both
visible cell-by-cell.

Colour: sequential single-hue (green) for magnitude-of-error with a neutral
grey for "nothing reported"; text is ink tokens only. Usage::

    python paper/fig_blind_goals.py results/eval_blind_flash.json results/eval_blind_pro.json
    # writes paper/fig_blind_goals.pdf and paper/fig_blind_goals.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _font import setup_font  # noqa: E402

setup_font()

INK = "#262522"
MUT = "#8a8782"
GRID = "#d9d7d3"
NA_COLOR = "#a8a39d"
NA = "·"

#: sequential green ramp for |relative error|, white at 0 (hit) -> deep green at
#: >=10x. CVD-safe single hue, validated lightness monotonic.
CMAP = LinearSegmentedColormap.from_list(
    "shear_err",
    ["#ffffff", "#cfe6d2", "#7fb98c", "#3a7d44", "#14432a"],
)
VMAX = 10.0  # clamp: 10x or worse all read as max green

#: cells are coloured on log(1+v), not v, so the realistic sub-2x misses still
#: occupy visible colour steps instead of a wash of near-white. The colourbar
#: uses the same norm, so the ticks line up with what the cells show.
ERR_NORM = mcolors.FuncNorm(
    (
        lambda v: np.log10(1 + np.clip(v, 0, VMAX)) / np.log10(1 + VMAX),
        lambda t: (1 + VMAX) ** t - 1,
    ),
    vmin=0, vmax=VMAX,
)

GOAL_LABELS = {
    "blind-liver-sinusoid": "liver sinusoid",
    "blind-kidney-ptec": "kidney PTEC",
    "blind-arterial-shear": "arterial",
    "blind-venular-shear": "venular",
    "blind-lung-alveolar": "lung alveolar",
    "blind-seed-hepg2-log": "HepG2 seeding",
    "blind-phh-seed": "PHH seeding",
    "blind-bbb-shear": "BBB",
    "blind-pulmonary-artery-shear": "pulmonary artery",
    "blind-gut-epithelial-shear": "gut epithelium",
    "blind-retinal-arteriole-shear": "retinal arteriole",
    "blind-lymphatic-shear": "lymphatic",
}

#: goals whose target is hinted (as a range) in the Labwright system prompt.
#: Marked with † so the cold-only recovery is countable from the figure.
PROMPT_BACKED = {"blind-liver-sinusoid", "blind-lung-alveolar", "blind-bbb-shear"}

def _val(rec: dict | None, key: str) -> float | None:
    if not rec:
        return None
    r = rec.get("recovery")
    if not r:
        return None
    if key in r:
        return r[key]
    # Two seeding goals carry only a seed_count recovery target; fall back to
    # their single recovery value so they render as errors, not "unverifiable".
    if len(r) == 1:
        return next(iter(r.values()))
    return None


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 1
    flash = json.load(open(argv[0]))
    pro = json.load(open(argv[1]))
    files = [flash, pro]
    gold = [pe["id"] for pe in flash["per_entry"]]

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(top=0.82, bottom=0.09, left=0.17, right=0.90)

    n_rows = len(gold)
    n_cols = 4  # flash-bare, flash-lab, pro-bare, pro-lab

    for ri, gid in enumerate(gold):
        for ci, (model_idx, sys_key) in enumerate(
            [(0, "bare"), (0, "labwright"), (1, "bare"), (1, "labwright")]
        ):
            entry = {pe["id"]: pe for pe in files[model_idx]["per_entry"]}.get(gid)
            rec = entry.get(sys_key) if entry else None
            v = _val(rec, "shear_pa")
            x = ci
            y = ri
            if v is None:
                ax.add_patch(plt.Rectangle((x - 0.45, y - 0.4), 0.9, 0.8,
                                           fc=NA_COLOR, ec="none", zorder=2))
                ax.text(x, y, NA, ha="center", va="center", fontsize=9,
                        color="white", zorder=4)
            else:
                vc = min(v, VMAX)
                ax.add_patch(plt.Rectangle((x - 0.45, y - 0.4), 0.9, 0.8,
                                           fc=CMAP(ERR_NORM(vc)), ec="none", zorder=2))
                # every non-hit cell carries its own error value so a near-miss
                # (say 7% off) is legible even though its colour is near-white;
                # hits get a √. (√ is Times-safe; ✓ is not in the face.)
                if v <= 0.05:
                    label = "√"
                elif v < 1.0:
                    label = f"{v:.2f}"
                elif v < 10.0:
                    label = f"{v:.1f}×"
                else:
                    label = f"{v:.0f}×"
                ax.text(x, y, label, ha="center", va="center", fontsize=8,
                        color=INK, zorder=4,
                        fontweight="bold" if v <= 0.05 else "normal")

    # row labels
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(
        [GOAL_LABELS.get(g, g) + ("†" if g in PROMPT_BACKED else "") for g in gold],
        fontsize=8.5, color=INK,
    )
    # column labels (model family on top, system under)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(["", "", "", ""])
    for ci, name in enumerate(["flash", "flash", "pro", "pro"]):
        ax.text(ci, -0.75, name, ha="center", va="top", fontsize=9.5, color=INK,
                fontweight="bold")
    for ci, sys_name in enumerate(["bare", "Lab", "bare", "Lab"]):
        ax.text(ci, -1.15, sys_name, ha="center", va="top", fontsize=8, color=MUT)

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("blind-set recovery per goal",
                 fontsize=10, color=INK, pad=22)

    # column grouping: a hairline between the flash and pro families, so the
    # two model families read as two blocks rather than four loose columns.
    ax.plot([1.5, 1.5], [-0.35, n_rows - 0.35], color=GRID, lw=1.0, zorder=1)

    # colourbar — same log norm as the cells, so ticks line up with colours
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=ERR_NORM)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("|relative recovery error| (clamped at 10×)",
                   fontsize=8, color=MUT)
    cbar.set_ticks([0, 0.5, 1, 5, 10])
    cbar.set_ticklabels(["hit", "0.5×", "1×", "5×", "≥10×"])
    cbar.ax.tick_params(labelsize=8, colors=MUT)

    # legend for NA + hit; the white swatch needs an edge to be visible on the
    # white figure background
    handles = [
        Patch(facecolor=CMAP(0.0), edgecolor=GRID, linewidth=0.8,
              label="√ recovery within 5% of target"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.90),
               ncol=2, frameon=False, fontsize=8)
    fig.text(0.02, 0.015,
             "† = target hinted in system prompt · the two seeding goals are scored on cell-count recovery",
             fontsize=8, color=MUT, ha="left", va="bottom")

    out = Path(__file__).resolve().parent
    fig.savefig(out / "fig_blind_goals.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(out / "fig_blind_goals.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {out / 'fig_blind_goals.pdf'} and {out / 'fig_blind_goals.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
