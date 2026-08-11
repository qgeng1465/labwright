"""Render the paper's reverse-verification figure (Fig 4).

Shows the six protocols run through ``verify_published_protocol``. Two
published protocols (kidney Jang et al., lung Huh et al.) and four synthetic
controls. Left panel: overall verdict per protocol (ok / review_required /
unverifiable). Right panel: per-field verdict matrix (consistent / discrepancy /
not-claimed) across the six derived flow numbers, so the reproducibility gap
is visible cell-by-cell — a claimed number is either re-derived, flagged, or
absent.

Status colours are reserved for their meaning: consistent=good (green),
discrepancy=serious (red), not-claimed=warning (amber), unverifiable=neutral
(gray). Text is ink tokens only.

Usage::

    python paper/fig_verify.py results/eval_verify_batch.json
    # writes paper/fig_verify.pdf and paper/fig_verify.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

INK = "#262522"
MUT = "#8a8782"
GRID = "#d9d7d3"
GOOD = "#3a7d44"    # status: consistent / ok
SERIOUS = "#b3261e"  # status: discrepancy
WARN = "#b26a00"     # status: not claimed
NEUTRAL = "#a8a39d"  # status: unverifiable
ROW_ALT = "#f4f2ee"

#: display names for the six derived flow numbers
FIELDS = [
    ("shear_pa", "shear"),
    ("reynolds", "Re"),
    ("pressure_drop_pa", "ΔP"),
    ("residence_time_s", "t_res"),
    ("channel_volume_ul", "V_ch"),
    ("mean_velocity_mms", "ū"),
]
FIELD_KEYS = [k for k, _ in FIELDS]
FIELD_LABELS = [lbl for _, lbl in FIELDS]

#: (id, display name, kind label)
PROTOCOLS = [
    ("kidney-jang-2013", "kidney PTEC\n(Jang 2013)", "published"),
    ("huh-2010-lung", "lung\n(Huh 2010)", "published"),
    ("control-positive-shear", "positive\ncontrol", "control"),
    ("control-unit-mix-error", "unit mix-up\n(dyn/cm² vs Pa)", "control"),
    ("control-reynolds-error", "Re 4 orders\noff", "control"),
    ("control-pressure-drop-error", "ΔP 10×\noff", "control"),
]


def verdict_color(v: str) -> str:
    return {"consistent": GOOD, "ok": GOOD,
            "discrepancy": SERIOUS,
            "not_claimed": WARN,
            "review_required": WARN,
            "unverifiable": NEUTRAL}.get(v, NEUTRAL)


def main(argv: list[str]) -> int:
    if len(argv) < 1:
        print(__doc__)
        return 1
    with open(argv[0]) as fh:
        data = json.load(fh)

    # per-protocol per-field verdict matrix
    matrix: dict[str, dict[str, str]] = {}
    overall: dict[str, str] = {}
    for p in data["protocols"]:
        pid = p["id"]
        overall[pid] = p["actual"]
        cells: dict[str, str] = {}
        for c in p.get("detail", {}).get("checks", []):
            cells[c["field"]] = c["verdict"]
        matrix[pid] = cells

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6),
                             constrained_layout=True, width_ratios=[1.25, 3.4])
    fig.patch.set_facecolor("white")

    # ---- left: overall verdict ----
    ax = axes[0]
    y_positions = list(range(len(PROTOCOLS)))
    for y, (pid, label, _kind) in zip(y_positions, PROTOCOLS):
        ax.add_patch(plt.Rectangle((-0.5, y - 0.45), 3.4, 0.9,
                                   fc=ROW_ALT, ec="none", zorder=0))
        ax.barh(y, 1.0, height=0.55, color=verdict_color(overall[pid]),
                zorder=3, edgecolor="none")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([lbl for _, lbl, _ in PROTOCOLS], fontsize=7.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("overall verdict", fontsize=8.5, color=INK, pad=4)
    ax.tick_params(length=0)

    # ---- right: per-field matrix ----
    ax = axes[1]
    for y, (pid, _label, _kind) in zip(y_positions, PROTOCOLS):
        ax.add_patch(plt.Rectangle((-0.5, y - 0.45), len(FIELDS) + 0.5, 0.9,
                                   fc=ROW_ALT, ec="none", zorder=0))
        for x, fk in enumerate(FIELD_KEYS):
            v = matrix.get(pid, {}).get(fk, "not_claimed")
            ax.add_patch(plt.Rectangle((x - 0.45, y - 0.34), 0.9, 0.68,
                                       fc=verdict_color(v), ec="none", zorder=3))
    ax.set_xlim(-0.5, len(FIELDS) - 0.5)
    ax.set_ylim(-0.5, len(PROTOCOLS) - 0.5)
    ax.set_xticks(range(len(FIELDS)))
    ax.set_xticklabels(FIELD_LABELS, fontsize=7.5, color=INK)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("per-field verdict", fontsize=8.5, color=INK, pad=4)
    ax.tick_params(length=0)

    # legend
    handles = [
        Patch(facecolor=GOOD, label="consistent"),
        Patch(facecolor=SERIOUS, label="discrepancy"),
        Patch(facecolor=WARN, label="not claimed"),
        Patch(facecolor=NEUTRAL, label="unverifiable"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=4, frameon=False, fontsize=8)

    out = Path(__file__).resolve().parent
    fig.savefig(out / "fig_verify.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(out / "fig_verify.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {out / 'fig_verify.pdf'} and {out / 'fig_verify.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
