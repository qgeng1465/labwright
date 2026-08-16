"""Render the paper's benchmark figure from the committed result JSONs.

Five gold sets side by side — the 24-reading set (every goal states the
target), the 15-blind set (no target stated), the 15-3D-spheroid set (a
third domain with fragmented 3D-culture conventions), the 14-plate-culture set
and the 14-perfused-PK set — as 3 × 5 small
multiples, one row per headline metric (self-consistent rate, usable rate,
hallucination rate). Within each panel the two model families
(deepseek-v4-flash, deepseek-v4-pro) are grouped; the five systems (bare-LLM,
soft-gate, self-verify, and the two Labwright front-ends) sit as
adjacent bars. Color follows the *system*: one categorical hue per system —
neutral stone, warm ochre, cool sage, deep academic blue for the Labwright
agent loop, lilac for the Labwright fast-path (the local fine-tuned extractor)
— so the texture channel (45° hatch on Labwright, crosshatch on the
fast-path) plus the legend keep identity readable in print
and for CVD.

The last bar is **Labwright's fast path**: the same gate (raw inputs → derive
via the same calculators → the same verifier), but the raw inputs are produced
by a fixed local Qwen2.5-1.5B LoRA fine-tuned extractor instead of the LLM
agent loop. It is model-independent, so its bars are identical under flash and
pro. Honesty note carried in the figure: it is trained on synthetic instances
across all 11 domains (reading targets reused from the reading gold set), so
the reading column is in-distribution and the spheroid / culture / PK columns
are close to it; the blind column is a mix, and the hand-written post-v1
domains (fig_benchmark does not draw them) are the true held-out set.

The blind set is where the honest boundary of the gate shows: self-consistency
stays high for Labwright while the usable rate collapses, and the naive
alternatives (soft-gate, self-verify) reach a usable design only on the handful
of single-step goals the goal itself hands over (the three no-choice 24-reading
goals; a few pure-geometry/lookup spheroid goals after the string-format
fairness fix) — so the figure carries the paper's central caveat visually.

Data source: each result file (``eval_flash.json`` / ``eval_pro.json`` /
``eval_blind_*.json`` / ``eval_spheroid_*.json``) already contains all five
systems after the post-fix re-run (the fine-tuned rows are merged into both
model files, since the extractor is model-independent), so the figure reads
straight from the per-entry records of one file per set × model — no
competitor-file merging. Numbers are recomputed by ``eval.report.derive()``
from the raw per-entry records, never re-typed.

Layout: the set headers live in a reserved band above the panels (not as axes
text at y>1, which collided with the first-row titles), and each panel keeps
its own metric title. Value labels are drawn inside/above bars with per-bar
logic so they
never overlap a neighbour.

Usage::

    python paper/fig_benchmark.py \\
        results/eval_flash.json results/eval_pro.json \\
        results/eval_blind_flash.json results/eval_blind_pro.json \\
        results/eval_spheroid_flash.json results/eval_spheroid_pro.json \\
        results/eval_culture_flash.json results/eval_culture_pro.json \\
        results/eval_pk_flash.json results/eval_pk_pro.json
    # writes paper/fig_benchmark.pdf and paper/fig_benchmark.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _font import setup_font  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.report import derive  # noqa: E402

setup_font()

# --- data ---
# Two model families, each carrying the four systems (bare / soft-gate /
# self-verify as a de-emphasized gray family, Labwright as the deep academic
# blue).
MODELS = ["deepseek-v4-flash", "deepseek-v4-pro"]
MODEL_SHORT = ["flash", "pro"]
#: (metric key, row title) — one row per headline metric.
METRICS = [
    ("self_consistent_rate", "Self-consistent rate"),
    ("usable_rate", "Usable rate"),
    ("hallucination_rate", "Hallucination rate"),
]
#: (column title, column subtitle). The reading set hands over the answer; the
#: blind set does not — that is the boundary the figure makes visible; the
#: spheroid set mixes a stated-goal arithmetic with fragmented 3D-culture
#: conventions (ULA / hanging-drop working volumes) that are domain knowledge;
#: the culture and PK sets add two further domains with their own calculators
#: and their own strictest-cross-check scoring.
SETS = [
    ("24-reading set", "target stated in the goal"),
    ("15-blind set", "no target stated"),
    ("15 3D-spheroid set", "3D-culture conventions"),
    ("14 plate-culture set", "plate conventions"),
    ("14 perfused-PK set", "single-compartment PK"),
]
#: (system key, legend label, bar color, edge/hatch color, hatch).
#: Categorical, one hue per system: a neutral light stone for the plain baseline,
#: a warm ochre for soft-gate, a cool sage for self-verify, the paper's deep
#: academic blue for Labwright, and a lilac for the local fine-tuned extractor.
#: Pairwise OKLab ΔE verified ≥ 15 normal-vision and ≥ 8 under protan simulation
#: (bare→soft 20.3, soft→self 16.6, self→lab 15.6; finetuned→others 15.2–21.3
#: normal, 13.4–26.8 protan). Labwright keeps the 45° hatch and finetuned the
#: crosshatch as extra identity channels.
SYSTEMS = [
    ("bare", "bare-LLM", "#C9C2B6", "none", None),
    ("soft_gate", "soft-gate", "#C07C2B", "#9A611F", "o"),
    ("self_verify", "self-verify", "#5F7668", "#3F5146", "+"),
    ("labwright", "Labwright", "#2E5598", "#1f3f70", "//"),
    ("finetuned", "Labwright fast-path", "#A080B0", "#6E4F8A", "x"),
]
INK = "#262522"          # text primary
MUT = "#8a8782"          # muted text (axis, sub-label)
GRID = "#d9d7d3"         # hairline grid
WHITE = "#ffffff"


def _label_color(hexc: str) -> str:
    """Dark text on light bars, white on dark bars — never a fixed label color."""
    c = [int(hexc[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lum = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    return WHITE if lum < 0.45 else INK
INK = "#262522"          # text primary
MUT = "#8a8782"          # muted text (axis, sub-label)
GRID = "#d9d7d3"         # hairline grid

N_SYSTEMS = len(SYSTEMS)


def _load_set(path: str) -> dict:
    """Read one set×model file and return its four-system derived metrics."""
    with open(path) as fh:
        return derive(json.load(fh))


def main(argv: list[str]) -> int:
    # argv: [reading-flash, reading-pro, blind-flash, blind-pro,
    #        spheroid-flash, spheroid-pro, culture-flash, culture-pro,
    #        pk-flash, pk-pro]
    if len(argv) < 10:
        print(__doc__)
        return 1
    sets = [
        [_load_set(argv[0]), _load_set(argv[1])],  # reading: flash, pro
        [_load_set(argv[2]), _load_set(argv[3])],  # blind: flash, pro
        [_load_set(argv[4]), _load_set(argv[5])],  # spheroid: flash, pro
        [_load_set(argv[6]), _load_set(argv[7])],  # culture: flash, pro
        [_load_set(argv[8]), _load_set(argv[9])],  # pk: flash, pro
    ]

    # Reserved top band for the set headers + legend so they never collide
    # with the first row of panels. Five columns × 3 rows; the width keeps each
    # panel as wide as the 3-column layout's panels (value labels are sized for
    # that width and overlap at any narrower value).
    fig, axes = plt.subplots(
        len(METRICS), len(SETS), figsize=(18.6, 5.1),
        sharey="row",
    )
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(top=0.72, bottom=0.09, left=0.08, right=0.985,
                        hspace=0.42, wspace=0.20)

    for col in range(len(SETS)):
        for row, (key, title) in enumerate(METRICS):
            ax = axes[row, col]
            data = sets[col]
            # 5 bars per model group; width sized so adjacent in-bar value
            # labels never graze (the old 0.62 pitch was narrower than the
            # widest label, so e.g. "100%" touched its neighbour's label).
            width = 0.80 / N_SYSTEMS
            offsets = [
                (i - (N_SYSTEMS - 1) / 2) * width for i in range(N_SYSTEMS)
            ]
            for i, model in enumerate(MODELS):
                if i >= len(data):
                    break
                d = data[i]
                pos = i
                for (sys_key, _label, color, edge, hatch), off in zip(SYSTEMS, offsets):
                    if sys_key not in d:  # a system missing from this run's file
                        continue
                    v = d[sys_key][key]
                    ax.bar(pos + off, v, width, color=color, edgecolor=edge,
                           hatch=hatch, zorder=3, linewidth=0.5)
                    # One label format (percent) for every bar — including 0%
                    # bars, so "nothing happened" is countable, not a gap.
                    # A tall bar sinks the label inside with a per-bar ink
                    # color, a short bar floats it above in dark ink.
                    txt = f"{100 * v:.0f}%"
                    if v > 0.30:
                        # Inside the bar, sunk far enough that two tall adjacent
                        # bars' labels never touch (5 bars/group is tight).
                        ax.text(pos + off, v - 0.03, txt, ha="center", va="top",
                                fontsize=7.5, color=_label_color(color))
                    else:
                        ax.text(pos + off, v + 0.015, txt, ha="center", va="bottom",
                                fontsize=8.0, color=INK)

            # axis dressing: recessive, ink-only
            ax.set_xticks(range(len(data)))
            ax.set_xticklabels(MODEL_SHORT[:len(data)], fontsize=8.5, color=INK)
            ax.set_ylim(0, 1.0)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8, color=MUT)
            ax.yaxis.grid(True, color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_color(GRID)
                spine.set_linewidth(0.6)
            ax.tick_params(length=0)
            ax.set_title(title, fontsize=9.5, color=INK, pad=3)

    # column headers sit in the reserved top band, centered on their panel
    # (axes positions, not hard-coded fractions — keeps both headers aligned
    # even when the panels are not symmetric about the figure centre).
    for col, (name, sub) in enumerate(SETS):
        pos = axes[0, col].get_position()
        cx = pos.x0 + pos.width / 2
        # Headers sit below the legend's rendered bbox (legend is centred on
        # the whole figure, so its left entries overlap a left-panel-centred
        # header unless the header is dropped far enough).
        fig.text(cx, 0.85, name, ha="center", va="bottom", fontsize=10.5,
                 color=INK, fontweight="bold")
        fig.text(cx, 0.80, sub, ha="center", va="top", fontsize=8, color=MUT)

    for ax in axes[:, 0]:
        ax.set_ylabel("fraction of goals", fontsize=8.5, color=MUT)

    handles = [
        Patch(facecolor=color, edgecolor=edge, hatch=hatch, linewidth=0.5, label=label)
        for (_key, label, color, edge, hatch) in SYSTEMS
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.95),
               ncol=len(SYSTEMS), frameon=False, fontsize=8.5)

    out = Path(__file__).resolve().parent
    fig.savefig(out / "fig_benchmark.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(out / "fig_benchmark.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {out / 'fig_benchmark.pdf'} and {out / 'fig_benchmark.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
