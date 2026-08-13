"""Post-render QA for fig_blind_goals + fig_scirecipe + fig_benchmark:
text-overlap census and rendered-pixel checks. Not a pytest module — run
directly:

    .venv/bin/python paper/_check_render.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

# capture the figure objects created inside each module's main()
_captured: dict[str, list] = {"figs": []}
_orig_subplots = plt.subplots


def _wrap(*a, **k):
    fig, ax = _orig_subplots(*a, **k)
    _captured["figs"].append(fig)
    return fig, ax


plt.subplots = _wrap


def _bboxes(fig):
    fig.canvas.draw()
    rnd = fig.canvas.get_renderer()
    boxes = []
    for art in fig.texts:
        boxes.append((art.get_text(), art.get_window_extent(rnd)))
    for ax in fig.axes:
        for art in ax.texts:
            boxes.append((art.get_text(), art.get_window_extent(rnd)))
        for art in ax.title.get_children() if hasattr(ax.title, "get_children") else []:
            boxes.append((art.get_text(), art.get_window_extent(rnd)))
        boxes.append((ax.title.get_text(), ax.title.get_window_extent(rnd)))
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                boxes.append((t.get_text(), t.get_window_extent(rnd)))
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            boxes.append((tick.get_text(), tick.get_window_extent(rnd)))
    return [(s, bb) for s, bb in boxes if s.strip()]


def _overlaps(fig, name):
    boxes = _bboxes(fig)
    hits = []
    n = len(boxes)
    for i in range(n):
        for j in range(i + 1, n):
            a = boxes[i][1]; b = boxes[j][1]
            if a is None or b is None:
                continue
            w = min(a.x1, b.x1) - max(a.x0, b.x0)
            h = min(a.y1, b.y1) - max(a.y0, b.y0)
            if w > 2 and h > 2:
                hits.append((boxes[i][0], boxes[j][0], round(w * h)))
    if hits:
        print(f"[{name}] OVERLAPS ({len(hits)}):")
        for s1, s2, area in hits:
            print(f"    {s1!r} <-> {s2!r}  ({area}px^2)")
    else:
        print(f"[{name}] 0 text overlaps")
    return len(hits)


def main() -> int:
    sys.path.insert(0, str(HERE))
    bad = 0

    fbg = importlib.import_module("fig_blind_goals")
    fbg.main(["results/eval_blind_flash.json", "results/eval_blind_pro.json"])
    bad += _overlaps(_captured["figs"][-1], "fig_blind_goals")

    fsc = importlib.import_module("fig_scirecipe")
    fsc.main(["results/eval_scirecipe_audit.json"])
    bad += _overlaps(_captured["figs"][-1], "fig_scirecipe")

    fbm = importlib.import_module("fig_benchmark")
    fbm.main([
        "results/eval_flash.json", "results/eval_pro.json",
        "results/eval_blind_flash.json", "results/eval_blind_pro.json",
        "results/eval_spheroid_flash.json", "results/eval_spheroid_pro.json",
        "results/eval_culture_flash.json", "results/eval_culture_pro.json",
        "results/eval_pk_flash.json", "results/eval_pk_pro.json",
    ])
    bad += _overlaps(_captured["figs"][-1], "fig_benchmark")

    fcmp = importlib.import_module("fig_model_compare")
    fcmp.main([
        "results/eval_flash.json", "results/eval_pro.json",
        "results/eval_k3.json", "results/eval_kimicode.json",
        "results/eval_blind_flash.json", "results/eval_blind_pro.json",
        "results/eval_blind_k3.json", "results/eval_blind_kimicode.json",
        "results/eval_spheroid_flash.json", "results/eval_spheroid_pro.json",
        "results/eval_spheroid_k3.json", "results/eval_spheroid_kimicode.json",
        "results/eval_culture_flash.json", "results/eval_culture_pro.json",
        "results/eval_culture_k3.json", "results/eval_culture_kimicode.json",
        "results/eval_pk_flash.json", "results/eval_pk_pro.json",
        "results/eval_pk_k3.json", "results/eval_pk_kimicode.json",
    ])
    bad += _overlaps(_captured["figs"][-1], "fig_model_compare")

    print(f"\nTOTAL overlaps: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
