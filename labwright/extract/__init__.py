"""Raw-input extractor: goal prose → raw design inputs → verified design.

The extractor is a small LoRA-tuned chat model (default Qwen2.5-1.5B-Instruct)
that turns an experimental goal into the *raw* input block of a
:class:`~labwright.schema.design.DesignPlan`. Derived numbers are never
proposed by the model: the pipeline feeds the extracted raw through
:func:`labwright.design.build_design` and :func:`labwright.verify.checker.verify_design`,
so every number the user sees was produced and re-proven by the deterministic
calculators. This package is the fine-tuned fast path of the same
raw→derive→verify contract the tool-using agent follows.
"""

from labwright.extract.data import SYSTEM_PROMPT, encode_example, raw_to_json
from labwright.extract.pipeline import Extractor, parse_json

__all__ = ["Extractor", "parse_json", "SYSTEM_PROMPT", "encode_example", "raw_to_json"]
