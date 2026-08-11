"""Structured output contracts for Labwright designs.

The agent returns a :class:`DesignPlan`; every derived numeric field is
recomputed and validated by :mod:`labwright.verify` before the plan is shown.
Schemas double as the JSON-Schema contract consumed by the demo, the SOP
renderer and downstream tools.
"""

from labwright.schema.design import (
    ChipGeometry,
    CellPlan,
    DerivedFlowMetrics,
    DesignPlan,
    DosePlan,
    FlowParams,
    StatsPlan,
)

__all__ = [
    "ChipGeometry",
    "FlowParams",
    "DerivedFlowMetrics",
    "CellPlan",
    "DosePlan",
    "StatsPlan",
    "DesignPlan",
]
