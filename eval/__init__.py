"""Benchmark harness for the Labwright paper.

Question: does constraining the LLM to *propose* (while calculators *compute*)
actually fix hallucinated wet-lab numbers?

Two systems are compared on a set of gold-standard experiments:

- **bare-LLM**: the model is asked to produce a full design JSON, including
  derived numbers, on its own.
- **Labwright**: the model proposes raw inputs; derived numbers come from the
  calculators and the verifier.

Metrics (computed here, nothing else):
  - parameter-recovery accuracy on shear stress / flow rate (relative error),
  - hallucination rate: fraction of derived numbers that fail the verifier.
"""

from eval.benchmark import evaluate, load_gold, GoldExperiment

__all__ = ["evaluate", "load_gold", "GoldExperiment"]
