"""Shared font setup for the paper figures.

All figure text renders in Times New Roman (metric-compatible serif) so the
charts match the manuscript body. The TNR face is not part of the stock
matplotlib bundle; it is registered from ``~/.fonts/times_new_roman.ttf``
(the mscorefonts Times New Roman Regular) at import time.

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

_FONT = Path.home() / ".fonts" / "times_new_roman.ttf"


def setup_font() -> None:
    """Register TNR and point the default family at it (no-op if missing)."""
    if _FONT.exists():
        try:
            fm.fontManager.addfont(str(_FONT))
        except Exception:
            pass
    available = {f.name for f in fm.fontManager.ttflist}
    if "Times New Roman" in available:
        matplotlib.rcParams["font.family"] = "Times New Roman"
        matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
        matplotlib.rcParams["mathtext.rm"] = "Times New Roman"
        matplotlib.rcParams["mathtext.it"] = "Times New Roman"
    # axes tick labels inherit the family; keep sizes explicit in each script
