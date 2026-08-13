"""Unit & dimension layer — the first verification layer above arithmetic.

Arithmetic cross-checks prove a number *follows from the inputs*; this layer
proves the number is *expressed in the right unit*. Every raw and derived field
carries a canonical unit (declared in :data:`CANONICAL_UNITS`, matching the
schema field names), and the alias table records the unit pairs that actually
bite in the wet lab and in the literature — most importantly ``dyn/cm^2`` vs
``Pa``, which differ by a factor of 10.

The killer case this layer exists for: a microfluidics paper or a lab protocol
states *wall shear stress in dyn/cm²* (e.g. "0.2 dyn/cm²"), and the model —
seeing a bare number and asked for Pa — reports **0.2 Pa**. That is ten times
the true value (0.02 Pa) and is not an arithmetic slip; it is a unit misread.
:func:`classify_unit_misread` turns "wrong number" into "right magnitude,
wrong unit", so the benchmark can report a *unit-misread rate* instead of
lumping the case into generic recovery error.
"""

from __future__ import annotations

from labwright.blocks import ALL_CANONICAL_UNITS
from labwright.calc.units import Q

#: Canonical unit of every raw and derived field, keyed by the field names used
#: in the schema and the verifier's issue records. This is the audit table:
#: a value is only correct when it is expressed in this unit. Declared once per
#: design domain in :mod:`labwright.blocks` and merged here; adding a domain's
#: units means editing that domain's ``Block``, not this table.
CANONICAL_UNITS: dict[str, str] = ALL_CANONICAL_UNITS

#: Unit aliases that matter in practice: ``(name, from_unit, to_unit, factor)``
#: where ``factor`` is the ratio a *claimed* value shows against the *true*
#: value when a number expressed in ``from_unit`` is reported as ``to_unit``:
#: ``claimed = source_in_from``, ``expected = source_in_from / factor``.
#: Equivalently ``factor = value_in_from / value_in_to = Q(1, to).to(from)`` —
#: the number of ``from`` units in one ``to`` unit. ``1 Pa = 10 dyn/cm^2``, so a
#: dyn/cm^2 number mislabelled Pa is 10× too big (``factor = 10``); a mL/min
#: number mislabelled µL/min is 1000× too small (``factor = 0.001``). Only
#: aliases whose *to* unit matches a field's canonical unit are considered for
#: that field, which kills most arithmetic false positives.
#: ``tests/test_units.py::test_alias_table_matches_pint`` asserts every factor
#: against pint, so the table cannot silently drift from the unit registry.
UNIT_ALIASES: list[tuple[str, str, str, float]] = [
    ("dyn/cm^2 read as Pa", "dyn/cm^2", "Pa", 10.0),
    ("Pa read as dyn/cm^2", "Pa", "dyn/cm^2", 0.1),
    ("mL/min read as uL/min", "mL/min", "uL/min", 0.001),
    ("uL/h read as uL/min", "uL/h", "uL/min", 60.0),
    ("cells/mm^2 read as cells/cm^2", "cells/mm^2", "cells/cm^2", 0.01),
    ("m/s read as mm/s", "m/s", "mm/s", 0.001),
    ("mm/s read as m/s", "mm/s", "m/s", 1000.0),
    ("min read as s", "min", "s", 1 / 60.0),
    ("s read as min", "s", "min", 60.0),
    ("uL read as mL", "uL", "mL", 1000.0),
    ("mL read as uL", "mL", "uL", 0.001),
    ("um read as mm", "um", "mm", 1000.0),
    ("mm read as um", "mm", "um", 0.001),
    ("ng/mL read as ug/mL", "ng/mL", "ug/mL", 1000.0),
]


def canonical_unit(field: str) -> str | None:
    """Canonical unit for a schema field, or ``None`` if undeclared."""
    return CANONICAL_UNITS.get(field)


def convert(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a value between two units via pint (dyn is defined in calc.units)."""
    return Q(value, from_unit).to(to_unit).magnitude


def classify_unit_misread(claimed: float | None, expected: float | None, field: str, tol: float = 0.20) -> dict | None:
    """Classify a claimed-vs-expected mismatch as a probable unit misread.

    Returns a record ``{"alias", "from", "to", "ratio", "expected_in_from"}``
    when ``claimed / expected`` is within ``tol`` of one of the alias factors
    whose *to* unit is the field's canonical unit; ``None`` otherwise.

    ``expected_in_from`` is the gold value expressed in the unit the model
    appears to have used — the number it *should* have reported, ready for a
    warning message. A ``None`` return means "no known unit pair explains this
    mismatch" — a genuine arithmetic or target error, not a unit misread.
    """
    if claimed is None or expected is None or expected == 0 or not (expected > 0):
        return None
    if not (claimed > 0) or claimed != claimed or expected != expected:
        return None
    unit = canonical_unit(field)
    if not unit:
        return None
    ratio = claimed / expected
    for name, from_unit, to_unit, factor in UNIT_ALIASES:
        if to_unit != unit:
            continue
        if abs(ratio - factor) <= tol * max(abs(factor), 1e-12):
            try:
                in_from = Q(expected, unit).to(from_unit).magnitude
            except Exception:  # noqa: BLE001 - unit not in pint registry
                in_from = expected * factor
            return {
                "alias": name,
                "from": from_unit,
                "to": to_unit,
                "ratio": round(ratio, 6),
                "expected_in_from": round(in_from, 6),
            }
    return None


__all__ = [
    "CANONICAL_UNITS",
    "UNIT_ALIASES",
    "canonical_unit",
    "convert",
    "classify_unit_misread",
]
