"""Render the boundary/adversarial fail-safe figure (reviewer demand #3).

The reviewer asks for boundary testing and active elicitation: adversarial
inputs (missing parameters, physical conflicts, lethal conditions), a measured
active-request-for-information / exception-catching success rate, and proof
that the system fails safe instead of fabricating.

Two panels over the 30 adversarial entries × 3 systems (bare-LLM /
code-interpreter / Labwright-with-elicitation) × 2 models:

* **Left — fail-safe vs fabrication**: ``fail_safe_rate``
  (elicit + reject + refuse, all divided by N) against ``fabrication_rate``
  per system. The story the reviewer asked for: Labwright's gate turns lethal
  and under-determined goals into elicitation or a verifier rejection, while
  the baselines that must answer from memory mostly fabricate.
* **Right — what each system did**: the per-entry outcome composition
  (elicit / reject / refuse / fabricate / code-error / no-answer), the "confusion
  matrix" for the boundary test.

All numbers are read from the committed ``results/adversarial_*.json`` summary
block — nothing is hard-coded, nothing is recomputed from the figure.

Usage::

    python paper/fig_failsafe.py results/adversarial_flash.json results/adversarial_pro.json
    # writes paper/fig_failsafe.pdf and paper/fig_failsafe.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _font import setup_font  # noqa: E402

setup_font()

MODELS = ["deepseek-v4-flash", "deepseek-v4-pro"]
MODEL_SHORT = ["flash", "pro"]
SYSTEMS = ["bare", "code_interpreter", "labwright"]
SYSTEM_LABEL = {"bare": "bare-LLM", "code_interpreter": "code-interp",
                "labwright": "Labwright"}
#: (outcome, label, color) for the stacked composition bars.
OUTCOMES = [
    ("elicit", "elicit", "#4E7A5A"),        # sage-green (ask, don't guess)
    ("reject", "verifier reject", "#2E5598"),  # deep blue (gate caught it)
    ("refuse", "refuse", "#7A8B99"),        # slate (declined)
    ("fabricate", "fabricate", "#C05B4A"),  # rust (the failure mode)
    ("code_error", "code error", "#8A5A9E"),  # violet
    ("no_answer", "no answer", "#C9C2B6"),  # stone
]
INK = "#262522"
MUT = "#8a8782"
GRID = "#d9d7d3"
WHITE = "#ffffff"


def _stack_counts(summary: dict, system: str) -> dict[str, int]:
    counts = summary["systems"][system]["outcome_counts"]
    return {o: counts.get(o, 0) for o, _l, _c in OUTCOMES}


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 1
    runs = {}
    for path, model in zip(argv[:2], MODELS):
        with open(path) as fh:
            runs[model] = json.load(fh)

    fig = plt.figure(figsize=(11.5, 4.4))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.30,
                          left=0.09, right=0.97, top=0.80, bottom=0.14)

    # ---- left: fail-safe vs fabrication -------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    metrics = [("fail_safe_rate", "fail-safe"), ("fabrication_rate", "fabricate")]
    width = 0.8 / len(SYSTEMS)
    offsets = [(i - (len(SYSTEMS) - 1) / 2) * width for i in range(len(SYSTEMS))]
    sys_color = {"bare": "#C9C2B6", "code_interpreter": "#C07C2B", "labwright": "#2E5598"}
    for m, model in enumerate(MODELS):
        summary = runs[model].get("summary", {})
        pos = m * (len(metrics) + 1)
        for mi, (key, label) in enumerate(metrics):
            for (s, off) in zip(SYSTEMS, offsets):
                if s not in summary.get("systems", {}):
                    continue
                v = summary["systems"][s].get(key, float("nan"))
                if v != v:
                    continue
                x = pos + mi * 0.35 + off * 0.8
                ax.bar(x, v, width * 0.8, color=sys_color[s], zorder=3)
                ax.text(x, v + 0.02, f"{100 * v:.0f}%", ha="center", va="bottom",
                        fontsize=6.8, color=INK)
    ax.set_xticks([0.5 + m * (len(metrics) + 1) for m in range(len(MODELS))])
    ax.set_xticklabels(MODEL_SHORT, fontsize=9, color=INK)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8, color=MUT)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(GRID); spine.set_linewidth(0.6)
    ax.tick_params(length=0)
    ax.set_title("Fail-safe vs fabrication (30 adversarial goals)", fontsize=10,
                 color=INK, pad=4)
    for m in range(len(MODELS)):
        base = m * (len(metrics) + 1)
        for mi, label in enumerate(["fail-safe", "fabricate"]):
            ax.text(base + mi * 0.35, -0.09, label, ha="center", va="top",
                    fontsize=6.8, color=MUT)
    handles = [Patch(facecolor=sys_color[s], edgecolor="none", label=SYSTEM_LABEL[s])
               for s in SYSTEMS]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.16),
              ncol=3, frameon=False, fontsize=8)

    # ---- right: outcome composition -----------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    x = 0
    xlabels = []
    for m, model in enumerate(MODELS):
        summary = runs[model].get("summary", {})
        for s in SYSTEMS:
            if s not in summary.get("systems", {}):
                continue
            counts = _stack_counts(summary, s)
            total = sum(counts.values())
            bottom = 0.0
            for out, label, color in OUTCOMES:
                frac = counts[out] / total if total else 0.0
                if frac <= 0:
                    continue
                ax2.bar(x, frac, 0.62, bottom=bottom, color=color,
                        edgecolor=WHITE, linewidth=0.4, zorder=3)
                bottom += frac
            ax2.text(x, bottom + 0.012, f"{total}", ha="center", va="bottom",
                     fontsize=7, color=INK)
            xlabels.append(f"{MODEL_SHORT[m]}\n{SYSTEM_LABEL[s]}")
            x += 1
        x += 0.35
    ax2.set_xticks(range(len(xlabels)))
    ax2.set_xticklabels(xlabels, fontsize=7.5, color=INK)
    ax2.set_ylim(0, 1.12)
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8, color=MUT)
    ax2.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax2.set_axisbelow(True)
    for spine in ax2.spines.values():
        spine.set_color(GRID); spine.set_linewidth(0.6)
    ax2.tick_params(length=0)
    ax2.set_title("Boundary-test outcome composition (confusion matrix)",
                  fontsize=10, color=INK, pad=4)
    handles = [Patch(facecolor=color, edgecolor="none", label=label)
               for _o, label, color in OUTCOMES]
    ax2.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.12),
               ncol=3, frameon=False, fontsize=7.5)

    out = Path(__file__).resolve().parent
    fig.savefig(out / "fig_failsafe.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(out / "fig_failsafe.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {out / 'fig_failsafe.pdf'} and {out / 'fig_failsafe.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
