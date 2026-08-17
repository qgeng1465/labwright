"""Render the LabMath-Bench TBA figure from the full ablation result JSONs.

The reviewer's headline metric is TBA — tolerance-bound accuracy,
:math:`\\mathrm{TBA}(\\tau) = \\frac{1}{N}\\sum_{(e,k)}\\mathbb{I}(|y_{pred}-y_{true}|/y_{true} \\le \\tau)`
over every scored (entry, gold-target) pair, at the strict ``τ = 0.05``
threshold. Two panels:

* **Left — TBA(0.05) by level**: the three reviewer levels (L1 fluid & spatial
  engineering, L2 biochemical stoichiometry, L3 pipeline parameterization), per
  model × per system (bare-LLM / code-interpreter / Labwright), with a Wilson
  score interval over the scored key-pairs as error bars. The honest reading:
  Labwright's numbers come from deterministic calculators, so TBA(0.05) ≈ 1
  wherever the model's *parameter extraction* was right — a miss is exactly a
  parameter-extraction failure, not a calculator error.
* **Right — TBA–τ curve**: TBA at ``τ ∈ {0.01, 0.05, 0.10, 0.25}`` across the
  whole combined dataset, recomputed inline from the per-entry recovery
  errors (never re-typed). The gap at every threshold is the answer to
  "does executing code or doing the arithmetic in your head beat verified
  calculators".

Data source: the full-ablation files ``results/eval_labmath_flash.json`` and
``results/eval_labmath_pro.json`` (bare / code_interpreter / labwright on all
610 combined gold entries). Numbers are recomputed from the raw per-entry
``recovery`` dicts; nothing is hard-coded.

Usage::

    python paper/fig_tba.py results/eval_labmath_flash.json results/eval_labmath_pro.json
    # writes paper/fig_tba.pdf and paper/fig_tba.png
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _font import setup_font  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.ci import wilson_ci  # noqa: E402

setup_font()

LEVELS = ["L1", "L2", "L3"]
LEVEL_LABEL = {
    "L1": "L1 fluid & spatial",
    "L2": "L2 stoichiometry",
    "L3": "L3 pipeline",
}
MODELS = ["deepseek-v4-flash", "deepseek-v4-pro"]
MODEL_SHORT = ["flash", "pro"]
TAUS = [0.01, 0.05, 0.10, 0.25]
SYSTEMS = [
    ("bare", "bare-LLM", "#C9C2B6", "none", None),
    ("code_interpreter", "code-interp", "#C07C2B", "#9A611F", "o"),
    ("labwright", "Labwright", "#2E5598", "#1f3f70", "//"),
]
INK = "#262522"
MUT = "#8a8782"
GRID = "#d9d7d3"
WHITE = "#ffffff"


def _tba_pairs(entries: list[dict], system: str) -> list[float]:
    """Per-key relative errors for one system across the run (the Bernoulli trials)."""
    errs: list[float] = []
    for e in entries:
        for err in e.get(system, {}).get("recovery", {}).values():
            if err == err:  # finite
                errs.append(err)
    return errs


def _tba_at(errs: list[float], tau: float) -> float:
    return (sum(1 for e in errs if e <= tau) / len(errs)) if errs else float("nan")


def _tba_by_level_at(entries: list[dict], system: str, tau: float) -> dict[str, tuple]:
    """(tba, n_pairs, [lo, hi]) per level at tolerance tau."""
    out: dict[str, tuple] = {}
    for level in LEVELS:
        pairs: list[float] = []
        for e in entries:
            gold = e.get("gold") or {}
            if gold.get("level") != level:
                continue
            for err in e.get(system, {}).get("recovery", {}).values():
                if err == err:
                    pairs.append(err)
        if not pairs:
            continue
        k = sum(1 for x in pairs if x <= tau)
        lo, hi = wilson_ci(k, len(pairs))
        out[level] = (k / len(pairs), len(pairs), lo, hi)
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 1
    runs = {}
    for path, model in zip(argv[:2], MODELS):
        with open(path) as fh:
            runs[model] = json.load(fh)

    fig = plt.figure(figsize=(11.5, 4.6))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.28,
                          left=0.09, right=0.97, top=0.82, bottom=0.14)

    # ---- left: TBA(0.05) by level ------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    width = 0.8 / len(SYSTEMS)
    offsets = [(i - (len(SYSTEMS) - 1) / 2) * width for i in range(len(SYSTEMS))]
    for m, model in enumerate(MODELS):
        entries = runs[model]["per_entry"]
        pos = m * (len(SYSTEMS) + 1)  # gap between model groups
        for (key, _label, color, edge, hatch), off in zip(SYSTEMS, offsets):
            by_level = _tba_by_level_at(entries, key, 0.05)
            x = pos + off
            for level in LEVELS:
                if level not in by_level:
                    continue
                tba, n, lo, hi = by_level[level]
                xl = x + (LEVELS.index(level) - 1) * 0.32
                ax.bar(xl, tba, width * 0.8, color=color, edgecolor=edge,
                       hatch=hatch, zorder=3, linewidth=0.5)
                # Clamp to >= 0: at tba=1.0 the Wilson upper bound can round
                # to 1.0 - 1e-16, making hi - tba a tiny negative float.
                ylo = max(0.0, tba - lo)
                yhi = max(0.0, hi - tba)
                ax.errorbar(xl, tba, yerr=[[ylo], [yhi]],
                            fmt="none", ecolor=INK, elinewidth=0.8,
                            capsize=2, zorder=4)
                ax.text(xl, tba + 0.015, f"{100 * tba:.0f}%", ha="center",
                        va="bottom", fontsize=6.8, color=INK)
    ax.set_xticks([0.5 + m * (len(SYSTEMS) + 1) for m in range(len(MODELS))])
    ax.set_xticklabels(MODEL_SHORT, fontsize=9, color=INK)
    ax.set_ylabel("TBA(τ=0.05)", fontsize=9, color=INK)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8, color=MUT)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(GRID); spine.set_linewidth(0.6)
    ax.tick_params(length=0)
    ax.set_title("TBA(τ=0.05) by level (Wilson CI over key-pairs)", fontsize=10,
                 color=INK, pad=4)
    # legend inside top-left (no set header band here)
    handles = [Patch(facecolor=c, edgecolor=e, hatch=h, linewidth=0.5, label=l)
               for (_k, l, c, e, h) in SYSTEMS]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.16),
              ncol=3, frameon=False, fontsize=8)

    # ---- right: TBA–τ curve -------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    for (key, _label, color, edge, hatch) in SYSTEMS:
        for m, model in enumerate(MODELS):
            entries = runs[model]["per_entry"]
            errs = _tba_pairs(entries, key)
            vals = [_tba_at(errs, tau) for tau in TAUS]
            style = "-" if m == 0 else "--"
            ax2.plot(TAUS, vals, style, color=color, marker="o", markersize=4,
                     linewidth=1.6, zorder=3)
    ax2.set_xlabel("tolerance τ", fontsize=9, color=INK)
    ax2.set_ylabel("TBA", fontsize=9, color=INK)
    ax2.set_xscale("log")
    ax2.set_xticks(TAUS)
    ax2.set_xticklabels([str(t) for t in TAUS], fontsize=8, color=INK)
    ax2.set_ylim(0, 1.05)
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8, color=MUT)
    ax2.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax2.set_axisbelow(True)
    for spine in ax2.spines.values():
        spine.set_color(GRID); spine.set_linewidth(0.6)
    ax2.tick_params(length=0)
    ax2.set_title("TBA–τ curve (full 610-entry set)", fontsize=10, color=INK, pad=4)
    # model legend as line-style swatches on the right
    line_handles = [
        plt.Line2D([0], [0], color="#666", linestyle="-", lw=1.6, label="flash"),
        plt.Line2D([0], [0], color="#666", linestyle="--", lw=1.6, label="pro"),
    ]
    ax2.legend(handles=line_handles, loc="lower right", frameon=False, fontsize=8)

    out = Path(__file__).resolve().parent
    fig.savefig(out / "fig_tba.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(out / "fig_tba.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {out / 'fig_tba.pdf'} and {out / 'fig_tba.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
