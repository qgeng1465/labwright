"""Render the ablation + confusion-matrix figure (reviewer demand #2).

The reviewer asks for the three-system ablation — Baseline A (pure LLM, bare),
Baseline B (LLM + code interpreter), and Labwright — with a confusion matrix
showing the calculation-error rate CER → 0. The outcome classes come straight
from the per-entry failure classification:

* ``ok`` — usable: self-consistent/verified and recovers every gold target.
* ``wrong_target`` — self-consistent but misses the gold (parameter/formula
  extraction wrong — the model picked the wrong magnitude or the wrong organ's
  number).
* ``calculation_error`` — produced numbers but internally inconsistent
  (a model's typed-from-memory or hand-written-code arithmetic that its own
  inputs do not reproduce).
* ``code_exec_error`` — the code-interpreter baseline: the program never ran
  to a clean RESULT (syntax error / exception / timeout) — the distinct
  "code never executed" cell.
* ``silence`` — produced nothing checkable.

Labwright's derived numbers come from deterministic calculators behind a
verifier gate, so a submitted plan is never internally inconsistent: the
``calculation_error`` and ``code_exec_error`` classes are structurally empty
for it (CER = 0). Its residual misses are ``wrong_target`` — the *parameter
extraction* NLU failure, which is precisely the remaining error Labwright can
have, never a calculation error.

Two panels:

* **Left — headline rates**: self-consistent rate, usable rate and TBA(τ=0.05)
  for the three systems × two models, recomputed by ``eval.report.derive``.
* **Right — confusion matrix**: per system × model, the stacked fraction of
  entries in each outcome class (the "what kind of error" breakdown). The
  calculation-error and code-exec-error bands collapse to zero for Labwright.

Data source: ``results/eval_labmath_flash.json`` / ``results/eval_labmath_pro.json``
(bare / code_interpreter / labwright on all 610 combined gold entries).

Usage::

    python paper/fig_ablation.py results/eval_labmath_flash.json results/eval_labmath_pro.json
    # writes paper/fig_ablation.pdf and paper/fig_ablation.png
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.report import derive  # noqa: E402

setup_font()

MODELS = ["deepseek-v4-flash", "deepseek-v4-pro"]
MODEL_SHORT = ["flash", "pro"]
SYSTEMS = ["bare", "code_interpreter", "labwright"]
SYSTEM_LABEL = {"bare": "bare-LLM", "code_interpreter": "code-interp",
                "labwright": "Labwright"}
#: (class key, label, color) for the confusion-matrix stacked bars.
CLASSES = [
    ("ok", "correct", "#5F7668"),                    # sage
    ("wrong_target", "wrong target", "#C07C2B"),     # ochre
    ("calculation_error", "calculation error", "#B45A4B"),  # rust
    ("code_exec_error", "code exec error", "#8A5A9E"),      # violet
    ("silence", "silence", "#C9C2B6"),               # stone
]
INK = "#262522"
MUT = "#8a8782"
GRID = "#d9d7d3"
WHITE = "#ffffff"


def _class_counts(entries: list[dict], system: str) -> dict[str, int]:
    out = {c: 0 for c, _l, _col in CLASSES}
    for e in entries:
        out[e.get(system, {}).get("failure", "silence")] = out.get(
            e.get(system, {}).get("failure", "silence"), 0) + 1
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 1
    results = {}
    for path, model in zip(argv[:2], MODELS):
        with open(path) as fh:
            results[model] = json.load(fh)

    fig = plt.figure(figsize=(11.5, 4.4))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.30,
                          left=0.09, right=0.97, top=0.80, bottom=0.14)

    # ---- left: headline rates ----------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    metrics = [("self_consistent_rate", "Self-consistent"), ("usable_rate", "Usable"),
               ("tba", "TBA(0.05)")]
    width = 0.8 / len(SYSTEMS)
    offsets = [(i - (len(SYSTEMS) - 1) / 2) * width for i in range(len(SYSTEMS))]
    sys_colors = {"bare": "#C9C2B6", "code_interpreter": "#C07C2B", "labwright": "#2E5598"}
    sys_hatch = {"bare": None, "code_interpreter": "o", "labwright": "//"}
    for m, model in enumerate(MODELS):
        d = derive(results[model])
        pos = m * (len(metrics) + 1)
        for mi, (key, _label) in enumerate(metrics):
            for (sys_key, off) in zip(SYSTEMS, offsets):
                if sys_key not in d:
                    continue
                v = d[sys_key].get(key, float("nan"))
                if v != v:
                    continue
                x = pos + mi * 0.35 + off * 0.8
                ax.bar(x, v, width * 0.8, color=sys_colors[sys_key],
                       edgecolor="none", hatch=sys_hatch[sys_key],
                       zorder=3, linewidth=0.5)
                ax.text(x, v + 0.015, f"{100 * v:.0f}%", ha="center",
                        va="bottom", fontsize=6.8, color=INK)
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
    ax.set_title("Headline rates (610-entry full set)", fontsize=10, color=INK, pad=4)
    # metric sub-labels under the model tick
    for m in range(len(MODELS)):
        base = m * (len(metrics) + 1)
        for mi, label in enumerate(["self-cons.", "usable", "TBA"]):
            ax.text(base + mi * 0.35, -0.09, label, ha="center", va="top",
                    fontsize=6.8, color=MUT)

    # ---- right: confusion matrix (outcome-class stacked bars) ---------------
    ax2 = fig.add_subplot(gs[0, 1])
    x = 0
    xlabels = []
    for m, model in enumerate(MODELS):
        entries = results[model]["per_entry"]
        for sys_key in SYSTEMS:
            counts = _class_counts(entries, sys_key)
            total = sum(counts.values())
            bottom = 0.0
            for cls, label, color in CLASSES:
                frac = counts[cls] / total if total else 0.0
                if frac <= 0:
                    continue
                ax2.bar(x, frac, 0.62, bottom=bottom, color=color,
                        edgecolor=WHITE, linewidth=0.4, zorder=3)
                bottom += frac
            ax2.text(x, bottom + 0.012, f"{total}", ha="center", va="bottom",
                     fontsize=7, color=INK)
            xlabels.append(f"{MODEL_SHORT[m]}\n{SYSTEM_LABEL[sys_key]}")
            x += 1
        x += 0.35  # gap between model groups
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
    ax2.set_title("Outcome classes (confusion matrix → CER)", fontsize=10,
                  color=INK, pad=4)
    handles = [Patch(facecolor=color, edgecolor="none", label=label)
               for _cls, label, color in CLASSES]
    ax2.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.12),
               ncol=len(CLASSES), frameon=False, fontsize=7.5)

    out = Path(__file__).resolve().parent
    fig.savefig(out / "fig_ablation.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(out / "fig_ablation.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {out / 'fig_ablation.pdf'} and {out / 'fig_ablation.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
