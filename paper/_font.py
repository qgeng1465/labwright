"""Shared font setup for the paper figures.

All figure text renders in the sans-serif convention expected for
Cell-Press-style figure lettering — Arial, with Helvetica / DejaVu Sans as
metric fallbacks — so the charts use the journal's non-serif standard
(Arial/Helvetica) rather than the serif manuscript body (Times New Roman).
The Arial family is not part of the stock matplotlib bundle; it is registered
from ``~/.fonts/Arial*.ttf`` (mscorefonts Arial Regular / Bold / Italic /
Bold-Italic) at import time. If Arial is absent the family list falls back to
Helvetica then DejaVu Sans, so the figures still render in a clean sans-serif
face.

Import *after* ``matplotlib.use("Agg")`` but before creating any figure::

    import matplotlib
    matplotlib.use("Agg")
    from _font import setup_font  # noqa: E402

    setup_font()
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
from matplotlib import font_manager as fm

#: (file, family) — best-effort registration from the user font directory.
_FILES = [
    ("Arial.ttf", "Arial"),
    ("Arial_Bold.ttf", "Arial"),
    ("Arial_Italic.ttf", "Arial"),
    ("Arial_Bold_Italic.ttf", "Arial"),
]
#: Preferred families in order; matplotlib walks the list and uses the first
#: one with a registered face (weights/styles resolve from the registered set).
_FAMILY = ["Arial", "Helvetica", "DejaVu Sans"]


def setup_font() -> None:
    """Register Arial (full family) and point the default family at it."""
    base = Path.home() / ".fonts"
    for name, _family in _FILES:
        path = base / name
        if path.exists():
            try:
                fm.fontManager.addfont(str(path))
            except Exception:
                pass
    available = {f.name for f in fm.fontManager.ttflist}
    resolved = [f for f in _FAMILY if f in available] or ["DejaVu Sans"]
    matplotlib.rcParams["font.family"] = resolved
    matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
    # axes tick labels inherit the family; keep sizes explicit in each script
