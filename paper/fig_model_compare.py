"""Cross-provider comparison: does the Labwright gate transfer across backends?

Supplementary figure to fig_benchmark. fig_benchmark is the paper's canonical
system comparison on the two DeepSeek backends; this figure instead keeps the
**system** fixed and varies the **backend**: for each of the five gold sets it
draws the Labwright usable rate (and, as the baseline reference, the bare-LLM
usable rate) for the four benchmarked model families —

* `deepseek-v4-flash` and `deepseek-v4-pro` (the main results),
* `k3` (Kimi Code),
* `kimi-for-coding` (Kimi Code).

The two panels make the paper's boundary condition visible at a glance: the
gate transfers to every backend that can reliably execute the tool loop (flash,
pro, k3 all land 87–100 % usable on the reading/spheroid/culture/PK sets), and
collapses for a backend that cannot (kimi-for-coding fixates on a malformed
tool argument and never calls `submit_design` — 0 % usable on reading and
blind). The bare panel shows the transfer is *not* the backend's baseline
doing the work: every backend's bare usable rate is near zero on the reading
set.

Honesty notes carried in the figure:
* kimi-for-coding's failure is the tool loop, not temperature: all Kimi runs
  used temperature 0.6 with thinking disabled, the DeepSeek runs 0.2 (the
  coding endpoint only accepts 1.0 on the plain path; our request shape
  accepted 0.6). A higher temperature cannot explain k3's high usable rate.
* Numbers are recomputed by ``eval.report.derive()`` from the committed
  per-entry records, never re-typed.

Data source: 5 sets × 4 backends =
``eval_{flash,pro,k3,kimicode}.json`` (reading),
``eval_blind_*``, ``eval_spheroid_*``, ``eval_culture_*``, ``eval_pk_*`` in
``results/``.

Usage::

    python paper/fig_model_compare.py \\
        results/eval_flash.json results/eval_pro.json results/eval_k3.json results/eval_kimicode.json \\
        results/eval_blind_flash.json results/eval_blind_pro.json results/eval_blind_k3.json results/eval_blind_kimicode.json \\
        results/eval_spheroid_flash.json results/eval_spheroid_pro.json results/eval_spheroid_k3.json results/eval_spheroid_kimicode.json \\
        results/eval_culture_flash.json results/eval_culture_pro.json results/eval_culture_k3.json results/eval_culture_kimicode.json \\
        results/eval_pk_flash.json results/eval_pk_pro.json results/eval_pk_k3.json results/eval_pk_kimicode.json
    # writes paper/fig_model_compare.pdf and paper/fig_model_compare.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _font import setup_font  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.report import derive  # noqa: E402

setup_font()

# --- data ---
SETS = [
    ("24-reading", "target stated"),
    ("15-blind", "no target"),
    ("15 spheroid", "3D culture"),
    ("14 culture", "plate"),
    ("14 PK", "compartment"),
]
# (short label, full model name, bar color). Four categorical hues, pairwise
# OKLab-ΔE-separated, chosen to keep flash/pro (the paper's two backends) in
# the blue family while k3 goes green and kimi-for-coding ochre — the failing
# backend is the warmest, reading instantly as "the outlier".
BACKENDS = [
    ("flash", "deepseek-v4-flash", "#2E5598"),
    ("pro", "deepseek-v4-pro", "#6A4FA3"),
    ("k3", "k3 (Kimi Code)", "#5F7668"),
    ("kimi-code", "kimi-for-coding (Kimi Code)", "#C07C2B"),
]
INK = "#262522"
MUT = "#8a8782"
GRID = "#d9d7d3"
WHITE = "#ffffff"


def _label_color(hexc: str) -> str:
    c = [int(hexc[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lum = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    return WHITE if lum < 0.45 else INK


def _load(path: str) -> dict:
    with open(path) as fh:
        return derive(json.load(fh))


def main(argv: list[str]) -> int:
    # argv: 5 sets × 4 backends, ordered reading→pk, each set flash,pro,k3,kimi-code
    if len(argv) < 20:
        print(__doc__)
        return 1
    per_set = []
    for i in range(5):
        block = argv[i * 4:(i + 1) * 4]
        per_set.append({key: _load(path) for (key, _n, _c), path in zip(BACKENDS, block)})

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.4))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(top=0.80, bottom=0.16, left=0.075, right=0.985, wspace=0.22)

    metrics = [
        ("labwright", "Labwright usable rate"),
        ("bare", "bare-LLM usable rate (reference)"),
    ]
    for ax, (sys_key, title) in zip(axes, metrics):
        n = len(BACKENDS)
        # spread the cluster across most of the set width; the 0.88 pitch keeps
        # adjacent above-bar value labels from grazing (0.72 was narrower than
        # the widest label, so equal-height neighbours touched).
        step = 0.88 / n
        width = step - 0.03  # visible gap between adjacent bars
        offsets = [(j - (n - 1) / 2) * step for j in range(n)]
        for si, (sname, _ssub) in enumerate(SETS):
            data = per_set[si]
            for (key, _full, color), off in zip(BACKENDS, offsets):
                v = data[key][sys_key]["usable_rate"]
                ax.bar(si + off, v, width, color=color, edgecolor=color,
                       zorder=3, linewidth=0.5)
                # One placement rule (label above the bar) so equal-height
                # neighbours and the 0.30 inside/above boundary can never
                # collide. Every bar gets a label, including 0 % bars — an
                # unlabelled zero-height bar reads as a missing datum rather
                # than the "0 % usable" claim it is (kimi-for-coding collapses
                # to 0 % on four sets).
                ax.text(si + off, v + 0.014, f"{100 * v:.0f}%", ha="center",
                        va="bottom", fontsize=7.0, color=INK)
        ax.set_xticks(range(len(SETS)))
        ax.set_xticklabels([s for s, _ss in SETS], fontsize=8.5, color=INK)
        ax.set_ylim(0, 1.06)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8, color=MUT)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color(GRID)
            spine.set_linewidth(0.6)
        ax.tick_params(length=0)
        ax.set_title(title, fontsize=9.5, color=INK, pad=3)

    axes[0].set_ylabel("fraction of goals", fontsize=8.5, color=MUT)
    axes[0].set_ylim(0, 1.06)
    axes[0].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[0].set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8, color=MUT)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=color, label=f"{short} — {full}")
        for (short, full, color) in BACKENDS
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=4, frameon=False, fontsize=8)

    out = Path(__file__).resolve().parent
    fig.savefig(out / "fig_model_compare.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(out / "fig_model_compare.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(out / "fig_model_compare.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {out / 'fig_model_compare.pdf'} and {out / 'fig_model_compare.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
