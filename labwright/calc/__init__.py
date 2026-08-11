"""Labwright calculators — deterministic, verifiable wet-lab mathematics.

Every number Labwright prints is computed here (or by a registered plugin),
never invented by the language model. Calculators are pure functions:
same inputs, same outputs, unit-tested against analytic solutions and
published literature values.

Submodules
----------
microfluidics : channel geometry, flow, shear stress, pressure, oxygen delivery
cell          : seeding, growth, expansion, confluence
dosing        : molarity, dilution, drug preparation
stats         : sample size, power, replicates, effect size
units         : pint-based unit registry and conversion helpers
"""

from labwright.calc import cell, dosing, microfluidics, stats  # noqa: F401
from labwright.calc.units import Q, ureg  # noqa: F401

__all__ = ["cell", "dosing", "microfluidics", "stats", "Q", "ureg"]
