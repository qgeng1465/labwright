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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _font import setup_font  # noqa: E402

setup_font()

INK = "#262522"
MUT = "#8a8782"
GRID = "#d9d7d3"
GOOD = "#3a7d44"    # status: consistent / ok
SERIOUS = "#b3261e"  # status: discrepancy
WARN = "#b26a00"     # status: not claimed
NEUTRAL = "#a8a39d"  # status: unverifiable
ROW_ALT = "#f1f0ed"

#: display names (with units) for the six derived flow numbers
FIELDS = [
    ("shear_pa", "shear (Pa)"),
    ("reynolds", "Re"),
    ("pressure_drop_pa", "ΔP (Pa)"),
    ("residence_time_s", "t_res (s)"),
    ("channel_volume_ul", "V_ch (µL)"),
    ("mean_velocity_mms", "ū (mm/s)"),
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


#: prose labels for the overall verdicts shown in the left panel
VERDICT_LABELS = {
    "ok": "verified",
    "review_required": "needs review",
    "unverifiable": "unverifiable",
}


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

    # No constrained_layout: it silently ignores the subplots_adjust below (which
    # would leave the legend overlapping the panels). Manual fractions instead.
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6), width_ratios=[1.25, 3.4])
    fig.patch.set_facecolor("white")

    # ---- left: overall verdict (categorical, not a fake bar scale) ----
    ax = axes[0]
    y_positions = list(range(len(PROTOCOLS)))
    for y, (pid, label, _kind) in zip(y_positions, PROTOCOLS):
        ax.add_patch(plt.Rectangle((-0.5, y - 0.45), 3.4, 0.9,
                                   fc=ROW_ALT, ec="none", zorder=0))
        # a verdict is a category, so it gets a categorical mark — a coloured
        # chip plus prose — not a bar whose length would pretend to be a number.
        ax.add_patch(plt.Rectangle((0.15, y - 0.20), 0.40, 0.40,
                                   fc=verdict_color(overall[pid]), ec="none", zorder=3))
        ax.text(0.68, y, VERDICT_LABELS[overall[pid]], ha="left", va="center",
                fontsize=8, color=INK, zorder=4)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([lbl for _, lbl, _ in PROTOCOLS], fontsize=8, color=INK)
    # Explicit ylim before invert, matching the right panel exactly. Without it
    # matplotlib adds a default 5% margin and the two panels' rows drift apart
    # (top row left-high/right-low, bottom row left-low/right-high).
    ax.set_ylim(-0.5, len(PROTOCOLS) - 0.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 2.0)
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("overall verdict", fontsize=9.5, color=INK, pad=4)
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
            # CVD channel: a discrepancy carries a × so colour-blind readers
            # read shape, not hue. (× is Times-safe; ✓ is not in the face.)
            if v == "discrepancy":
                ax.text(x, y, "×", ha="center", va="center", fontsize=12,
                        color="#ffffff", fontweight="bold", zorder=5)
    ax.set_xlim(-0.5, len(FIELDS) + 1.4)
    ax.set_ylim(-0.5, len(PROTOCOLS) - 0.5)
    # invert so row 0 (the first protocol) is at the top, matching the left
    # panel — otherwise the two panels' rows run in opposite order and the
    # matrix reads as the wrong protocol having each discrepancy.
    ax.invert_yaxis()
    ax.set_xticks(range(len(FIELDS)))
    ax.set_xticklabels(FIELD_LABELS, fontsize=8, color=INK)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("per-field verdict", fontsize=9.5, color=INK, pad=4)
    ax.tick_params(length=0)

    # separator between the published protocols and the synthetic controls
    for panel in axes:
        panel.plot([-0.5, panel.get_xlim()[1]], [1.5, 1.5], color=GRID, lw=1.0,
                   zorder=2, clip_on=False)
    # group labels, rotated at the far right of the matrix
    for y, name in ((0.5, "published protocols"), (3.5, "synthetic controls")):
        ax.text(len(FIELDS) + 0.4, y, name, ha="left", va="center",
                rotation=90, fontsize=8, color=MUT)

    # one legend for both panels: the left verdicts and the right cells share
    # the same status colours (ok = consistent, review-needed = amber).
    handles = [
        Patch(facecolor=GOOD, label="consistent / ok"),
        Patch(facecolor=SERIOUS, label="discrepancy"),
        Patch(facecolor=WARN, label="not claimed / needs review"),
        Patch(facecolor=NEUTRAL, label="unverifiable"),
    ]
    fig.subplots_adjust(top=0.82)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.93),
               ncol=4, frameon=False, fontsize=8.5)

    out = Path(__file__).resolve().parent
    fig.savefig(out / "fig_verify.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(out / "fig_verify.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(out / "fig_verify.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {out / 'fig_verify.pdf'} and {out / 'fig_verify.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
