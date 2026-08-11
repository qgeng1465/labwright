"""Render the paper's benchmark figure from the committed result JSONs.

Three-panel small multiples, one panel per headline metric (self-consistent
rate, usable rate, hallucination rate). Within each panel the two model
families (deepseek-v4-flash, deepseek-v4-pro) are grouped; the two systems
(bare-LLM vs Labwright) sit as adjacent bars. Color follows the *system* —
bare-LLM is always slot-1 blue, Labwright always slot-2 orange — never the
rank, so it stays constant across panels.

Palette: the dataviz default categorical slots 1/2, validated in both modes
against the documented light/dark surfaces (normal-vision ΔE >= 15 and CVD
ΔE >= 8 both clear; contrast >= 3:1). Text uses ink tokens only — series
colors never carry text.

Usage::

    python paper/fig_benchmark.py results/eval_flash.json results/eval_pro.json
    # writes paper/fig_benchmark.pdf and paper/fig_benchmark.png

Numbers come from eval/report.derive(), which recomputes every metric from
per-entry records — the same source as Table 1.
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
MODELS = ["deepseek-v4-flash", "deepseek-v4-pro"]
METRICS = [
    ("self_consistent_rate", "Self-consistent rate", "higher is better"),
    ("usable_rate", "Usable rate", "higher is better"),
    ("hallucination_rate", "Hallucination rate", "lower is better"),
]

BAR = "#b0ada8"          # bare-LLM — de-emphasis gray (identity via legend/labels)
LW = "#eb6834"           # Labwright — categorical slot 2 (orange), validated
LW_HATCH = "#c85a22"     # darker step of the orange ramp — the texture channel for print/CVD
INK = "#262522"          # text primary
MUT = "#8a8782"          # muted text (axis, sub-label)
GRID = "#d9d7d3"         # hairline grid

BARE_LABEL = "bare LLM"
LW_LABEL = "Labwright"


def load(path: str) -> tuple[str, dict]:
    with open(path) as fh:
        result = json.load(fh)
    return result.get("model", "?"), derive(result)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 1
    data: dict[str, dict] = {}
    for model, p in zip(MODELS, argv[:2]):
        data[model] = load(p)[1]

    fig, axes = plt.subplots(
        1, len(METRICS), figsize=(6.4, 2.55), constrained_layout=True, sharey=True
    )
    fig.patch.set_facecolor("white")

    for ax, (key, title, sub) in zip(axes, METRICS):
        groups = MODELS
        width = 0.30
        offsets = (-width / 2 - 0.01, width / 2 + 0.01)  # 2 px surface gap
        for i, model in enumerate(groups):
            d = data[model]
            bare_v = d["bare"][key]
            lw_v = d["labwright"][key]
            pos = i
            for off, v, color, ec, hatch in (
                (offsets[0], bare_v, BAR, "none", None),
                (offsets[1], lw_v, LW, LW_HATCH, "//"),
            ):
                ax.bar(
                    pos + off, v, width, color=color, edgecolor=ec,
                    hatch=hatch, zorder=3, linewidth=0.5,
                )
                if key != "hallucination_rate":
                    txt = f"{100*v:.0f}%"
                    anchor = "top" if v > 0.30 else "bottom"
                    ha = "center"
                    if anchor == "top":
                        ax.text(pos + off, v - 0.015, txt, ha=ha, va="top",
                                fontsize=7.5, color="#ffffff" if color != BAR else INK)
                    else:
                        ax.text(pos + off, v + 0.015, txt, ha=ha, va="bottom",
                                fontsize=7.5, color=INK)
                else:
                    # hallucination: 0.0 is the win; small values label below the line
                    txt = f"{v:.3f}" if v > 0 else "0.000"
                    if v > 0:
                        ax.text(pos + off, v + 0.015, txt, ha="center", va="bottom",
                                fontsize=7.5, color=INK)
                    else:
                        ax.text(pos + off, 0.015, txt, ha="center", va="bottom",
                                fontsize=7.5, color=INK)

        # axis dressing: recessive, ink-only
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(["flash", "pro"], fontsize=8, color=INK)
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

    n_gold = next(iter(data.values()))["n_gold"]
    axes[0].set_ylabel(f"fraction of {n_gold} gold goals", fontsize=8, color=MUT)

    handles = [
        Patch(facecolor=BAR, edgecolor="none", label=BARE_LABEL),
        Patch(facecolor=LW, edgecolor=LW_HATCH, hatch="//", linewidth=0.5, label=LW_LABEL),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=2, frameon=False, fontsize=8.5)

    out = Path(__file__).resolve().parent
    fig.savefig(out / "fig_benchmark.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_benchmark.png", dpi=300, bbox_inches="tight")
    print(f"wrote {out / 'fig_benchmark.pdf'} and {out / 'fig_benchmark.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
