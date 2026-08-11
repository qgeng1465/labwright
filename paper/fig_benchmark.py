"""Render the paper's benchmark figure from the committed result JSONs.

Two gold sets side by side — the 24-reading set (every goal states the target)
and the 12-blind set (no target stated) — as 3 × 2 small multiples, one row per
headline metric (self-consistent rate, usable rate, hallucination rate). Within
each panel the two model families (deepseek-v4-flash, deepseek-v4-pro) are
grouped; the four systems (bare-LLM, soft-gate, self-verify, Labwright) sit as
adjacent bars. Color follows the *system*: the three LLM-memory systems are a
de-emphasized gray family and Labwright is the saturated orange, so the texture
channel (45° hatch on Labwright) plus the legend keep identity readable in
print and for CVD.

The blind set is where the honest boundary of the gate shows: self-consistency
stays high for Labwright while the usable rate collapses, and the naive
alternatives (soft-gate, self-verify) never reach a usable design at all — so
the figure carries the paper's central caveat visually.

Palette: validated with ``scripts/validate_palette.js`` (light mode) — the gray
family is lightness-stepped with CVD-separated adjacent pairs, and the orange
keeps its documented contrast. Text uses ink tokens only — series colors never
carry text.

Usage::

    python paper/fig_benchmark.py \\
        results/eval_flash.json results/eval_flash.json \\
        results/eval_pro.json results/eval_pro.json \\
        results/eval_blind_flash.json results/eval_blind_flash.json \\
        results/eval_blind_pro.json results/eval_blind_pro.json \\
        results/eval_thoth.json results/eval_blind_thoth.json
    # writes paper/fig_benchmark.pdf and paper/fig_benchmark.png

After the post-fix re-run the soft-gate + self-verify rows live in the same
file as bare + Labwright, so the main and comp args point at the same file.
The optional final two args add a thoth-8b group (three memory-system bars, no
Labwright bar) from the prompt-only Thoth run. Numbers come from
eval/report.derive(), which recomputes every metric from per-entry records.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.report import derive  # noqa: E402

# --- data ---
# A third group (thoth-8b) is added when the optional Thoth result files are
# passed (argv[8] = reading, argv[9] = blind). Thoth is run through the bare
# harness only — it has no Labwright mode — so its group carries just the three
# memory-system bars.
MODELS = ["deepseek-v4-flash", "deepseek-v4-pro", "thoth-8b"]
MODEL_SHORT = ["flash", "pro", "thoth-8b"]
#: (metric key, row title, row subtitle) — one row per headline metric.
METRICS = [
    ("self_consistent_rate", "Self-consistent rate", "higher is better"),
    ("usable_rate", "Usable rate", "higher is better"),
    ("hallucination_rate", "Hallucination rate", "lower is better"),
]
#: (column title, column subtitle). The reading set hands over the answer; the
#: blind set does not — that is the boundary the figure makes visible.
SETS = [
    ("24-reading set", "target stated in the goal"),
    ("12-blind set", "no target stated"),
]
#: (system key, legend label, bar color, edge/hatch color, hatch).
#: Gray family re-stepped so the minimum OKLab pair distance clears the
#: normal-vision floor (min ΔE = 17.1); Labwright carries the saturated orange.
SYSTEMS = [
    ("bare", "bare-LLM", "#a8a39d", "none", None),
    ("soft_gate", "soft-gate", "#e0dcd5", "#a8a39d", "o"),
    ("self_verify", "self-verify", "#74706a", "#a8a39d", "+"),
    ("labwright", "Labwright", "#eb6834", "#c85a22", "//"),
]
INK = "#262522"          # text primary
MUT = "#8a8782"          # muted text (axis, sub-label)
GRID = "#d9d7d3"         # hairline grid

N_SYSTEMS = len(SYSTEMS)


def load_merged(main_path: str, comp_path: str) -> dict:
    """Merge bare+Labwright (main file) with soft-gate + self-verify (comp file).

    Bare-LLM comes from the main file so its numbers match the paper's Table 1
    run; soft-gate and self-verify come from the competitor batch.
    """
    with open(main_path) as fh:
        main = derive(json.load(fh))
    with open(comp_path) as fh:
        comp = derive(json.load(fh))
    merged = {key: val for key, val in main.items() if key in ("bare", "labwright")}
    merged["soft_gate"] = comp["soft_gate"]
    merged["self_verify"] = comp["self_verify"]
    merged["n_gold"] = main["n_gold"]
    return merged


def load_thoth(path: str) -> dict:
    """A Thoth group: derive the three memory systems from a single file."""
    with open(path) as fh:
        return derive(json.load(fh))


def main(argv: list[str]) -> int:
    # argv layout: [reading-main flash, reading-comp flash, reading-main pro,
    # reading-comp pro, blind-main flash, blind-comp flash, blind-main pro,
    # blind-comp pro, thoth-reading (optional), thoth-blind (optional)]
    if len(argv) < 8:
        print(__doc__)
        return 1
    pairs = [
        (argv[0], argv[1]),  # reading flash
        (argv[2], argv[3]),  # reading pro
        (argv[4], argv[5]),  # blind flash
        (argv[6], argv[7]),  # blind pro
    ]
    sets = [
        [load_merged(*pairs[0]), load_merged(*pairs[1])],  # reading: flash, pro
        [load_merged(*pairs[2]), load_merged(*pairs[3])],  # blind: flash, pro
    ]
    if len(argv) >= 10 and argv[8] and argv[9]:
        sets[0].append(load_thoth(argv[8]))  # reading: thoth-8b
        sets[1].append(load_thoth(argv[9]))  # blind: thoth-8b

    fig, axes = plt.subplots(
        len(METRICS), len(SETS), figsize=(8.6, 4.6),
        constrained_layout=True, sharey="row",
    )
    fig.patch.set_facecolor("white")

    for col in range(len(SETS)):
        for row, (key, title, sub) in enumerate(METRICS):
            ax = axes[row, col]
            data = sets[col]
            # 4 bars per model group; width sized so groups don't collide.
            width = 0.62 / N_SYSTEMS
            offsets = [
                (i - (N_SYSTEMS - 1) / 2) * width for i in range(N_SYSTEMS)
            ]
            for i, model in enumerate(MODELS):
                if i >= len(data):
                    break
                d = data[i]
                pos = i
                for (sys_key, _label, color, edge, hatch), off in zip(SYSTEMS, offsets):
                    if sys_key not in d:  # e.g. thoth-8b has no Labwright bar
                        continue
                    v = d[sys_key][key]
                    ax.bar(pos + off, v, width, color=color, edgecolor=edge,
                           hatch=hatch, zorder=3, linewidth=0.5)
                    if key != "hallucination_rate":
                        txt = f"{100*v:.0f}%" if v >= 0.005 else ""
                        if v > 0.30:
                            ax.text(pos + off, v - 0.015, txt, ha="center", va="top",
                                    fontsize=6.5, color="#ffffff" if color != "#b0ada8" else INK)
                        elif v >= 0.005:
                            ax.text(pos + off, v + 0.015, txt, ha="center", va="bottom",
                                    fontsize=6.5, color=INK)
                    else:
                        txt = f"{v:.3f}" if v > 0 else ""
                        if v > 0:
                            ax.text(pos + off, v + 0.015, txt, ha="center", va="bottom",
                                    fontsize=6.5, color=INK)
                        # v == 0.0: the win is self-evident, skip the label

            # axis dressing: recessive, ink-only
            ax.set_xticks(range(len(data)))
            ax.set_xticklabels(MODEL_SHORT[:len(data)], fontsize=8, color=INK)
            ax.set_ylim(0, 1.0)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=7.5, color=MUT)
            ax.yaxis.grid(True, color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_color(GRID)
                spine.set_linewidth(0.6)
            ax.tick_params(length=0)
            ax.set_title(f"{title}\n{sub}", fontsize=8.5, color=INK, pad=4)

    # column headers name the set and make its subtitle explicit
    for col, (name, sub) in enumerate(SETS):
        axes[0, col].text(
            0.5, 1.06, name, transform=axes[0, col].transAxes,
            ha="center", va="bottom", fontsize=9.5, color=INK, fontweight="bold",
        )
        axes[1, col].set_xlabel(sub, fontsize=7.5, color=MUT, labelpad=2)

    # each panel's denominator is its own set size (24 or 12), so the label
    # stays set-agnostic.
    for ax in axes[:, 0]:
        ax.set_ylabel("fraction of goals", fontsize=8, color=MUT)

    handles = [
        Patch(facecolor=color, edgecolor=edge, hatch=hatch, linewidth=0.5, label=label)
        for (_key, label, color, edge, hatch) in SYSTEMS
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=len(SYSTEMS), frameon=False, fontsize=8)

    out = Path(__file__).resolve().parent
    fig.savefig(out / "fig_benchmark.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_benchmark.png", dpi=300, bbox_inches="tight")
    print(f"wrote {out / 'fig_benchmark.pdf'} and {out / 'fig_benchmark.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
