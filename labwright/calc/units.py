"""Unit handling for Labwright.

pint is used for explicit, auditable unit conversion. The public calculator
functions accept plain floats in the units documented in their docstrings,
but every derived quantity can be converted back and forth through
:class:`pint.Quantity`.

Example
-------
>>> from labwright.calc.units import Q
>>> shear = Q(0.05, "Pa")
>>> shear.to("dyn/cm**2").magnitude  # OOC groups often quote dyn/cm^2
0.5
"""

from pint import Quantity, UnitRegistry

ureg = UnitRegistry()
ureg.define("dyn = 1e-5 N")  # dyne, common in microfluidics literature

Q = Quantity

__all__ = ["ureg", "Q"]
