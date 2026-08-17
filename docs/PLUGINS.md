# Labwright Plugin API — adding a deterministic calculator

Labwright's rule is *"the calculator is the knowledge base"*: every number the
agent emits comes from a pure, deterministic function that the verifier can
re-run. If your lab has a quantity you want Labwright to reason about (a new
assay, a plate format, a chip geometry, a software-pipeline parameter), you add
a **calculator** and a **Tool**. Everything else — the agent loop, the
verifier, the SOP provenance, the benchmark — picks it up automatically.

This document is the public contract for that extension point. It mirrors what
the code at [labwright/tools.py](../labwright/tools.py) already says in its
module docstring: the registry is the only bridge between the calculators and
the agent.

---

## 1. The core extension point: `register_tool`

Everything starts in [labwright/tools.py](../labwright/tools.py):

```python
from labwright.tools import register_tool

class MyQuantityParams(BaseModel):
    flow_rate_uLmin: float = Field(gt=0, description="Volumetric flow rate in µL/min")
    # ... pydantic validates input and produces the JSON Schema the LLM sees

def my_quantity(flow_rate_uLmin: float, ...) -> float:
    """Pure math. No I/O, no randomness, no network — must be re-runnable."""
    return ...

register_tool(
    MyQuantityParams,
    name="my_quantity",
    description="Compute my quantity from the flow rate and channel geometry.",
    func=my_quantity,
    category="fluidics",       # groups the tool in the agent's tool list
    units_out="Pa",            # human-readable unit label for docs/UI
)
```

That is the *minimum* viable plugin: a `Tool` binds a pydantic parameter model,
a pure calc function and prose. The agent can now call it, the JSON Schema is
generated from the model, and `list_tools()` / `tools_for_llm()` expose it to
the LLM and the demo — nothing else changes.

### Contract on the function

- **Pure**: same inputs → same output, no hidden state. The verifier re-runs
  your function with the agent's own inputs to check a claimed number; if the
  function is not reproducible, the check is meaningless.
- **Units in the signature**: `*_um`, `*_ml`, `*_mM` — never "the units are
  obvious". Field names carry units so the LLM and the user cannot misread a
  value.
- **Fail loudly**: raise `ValueError` on physically impossible input
  (negative concentration, zero volume, a channel taller than it is wide).
  The tool-call machinery turns that into a structured error the agent can
  read and correct — it is *not* a crash.

---

## 2. Making a number *verifiable* (the full lifecycle)

A calculator that is only callable as a Tool lets the agent produce numbers;
making those numbers *verified* against a design requires wiring the quantity
through the design pipeline. The reviewer-facing guarantee — "Labwright's
numbers are computed and re-proved, never guessed" — lives here.

The full lifecycle for a new quantity is:

| Step | File | What you add |
|------|------|--------------|
| 1. Calc | `labwright/calc/<domain>.py` | the pure function(s), with a docstring citing the formula and constants |
| 2. Tool | `labwright/tools.py` | `register_tool(...)` as above |
| 3. Block | `labwright/blocks.py` | a `Block` declaring `raw_keys`, `derived_keys`, `consistency_keys`, `field_map`, `canonical_units`, `sanity_bands` |
| 4. Schema | `labwright/design.py` (`DesignInput`) | a raw-input sub-dict so the agent can submit the raw parameters |
| 5. Derive | `labwright/design.py` (`build_design`) | `d["<derived>"] = calc.<fn>(...)` from the raw dict |
| 6. Sanity | `labwright/verify/sanity.py` + the Block's `sanity_bands` | `Band(soft_min, soft_max, hard_min, hard_max, description, units)` |
| 7. Gold | `eval/gold_*.json` | a ground-truth entry whose `expected` values come from the *same* calculator (never hand-typed) |
| 8. Tests | `labwright/calc/test_<domain>.py` | unit tests pinning the formula and the source string |

### 2a. Block — what the verifier understands

```python
Block(
    name="flow",
    plan_field="flow",              # key on DesignPlan
    input_field="flow",             # key on DesignInput
    calc=mf,                        # module whose functions are re-run
    raw_keys=("flow_rate_uLmin", "viscosity_pas", "density_kgm3"),
    derived_keys=("shear_pa", "reynolds", "pressure_drop_pa", ...),
    consistency_keys=...,           # derived keys the agent is asked to report
    field_map={...},                # gold target name -> plan field
    sanity_bands={
        "derived.shear_pa": Band(0.001, 10, 1e-4, 50, "Pa", "physiological wall shear"),
        ...
    },
    canonical_units={"shear_pa": "Pa", "flow_rate_uLmin": "µL/min"},
)
```

`consistency_keys` + `field_map` are what make a number *recoverable*: the
benchmark asks the agent for exactly those keys, and `relative_error` is
measured against the gold. `sanity_bands` are a *soft* (warning) / *hard*
(error) envelope; a hard violation makes `submit_design` return
`review_required` and the gate refuses the number.

### 2b. Bands: the fail-safe envelope

```python
Band(soft_min, soft_max, hard_min, hard_max, description, units)
```

- Inside `[soft_min, soft_max]` → fine.
- Outside the soft range but inside `[hard_min, hard_max]` → warning ("safety
  hint"); the design is accepted if the agent explains it in `caveats`.
- Outside `[hard_min, hard_max]` → **error**; the verifier rejects the design.

These bands are what the *boundary evaluation* (reviewer demand #3) exercises:
a lethal shear, an impossible geometry, a toxic DMSO load all land outside a
hard band and the gate refuses them. Add a band for every quantity you ship —
a number with no envelope is not fail-safe.

### 2c. Why derived keys are never hand-typed

The `_reject_derived_fields` check in `submit_design` *rejects* any submitted
dict that already contains a derived key. The agent literally cannot write
`shear_pa: 0.05` — it must submit raw inputs and let `build_design` compute
`shear_pa`. This is the mechanism that makes calculation-error rate = 0 by
construction for Labwright.

---

## 3. What a plugin gets for free

Once you add the Tool and (ideally) the Block:

- **Agent loop** — the tool is callable; the LLM can reason with it.
- **Verifier** — derived numbers are re-computed and band-checked on every
  `submit_design`.
- **SOP provenance** — `labwright/sop/provenance.py::provenance_for` renders
  your quantity as a {formula, inputs, units, value, status} record, which
  becomes the field-level DAG in `paper/fig_protocol_dag.py` and the
  supplementary traceability logs.
- **Benchmark** — add `name` to `eval/run_benchmark.py --systems` and your
  gold's targets to a `GoldExperiment`; the harness scores it with the same
  relative-error, TBA and Wilson-CI machinery as every other domain.
- **Fail-safe eval** — add a `physical_conflict` / `lethal_condition` entry to
  `eval/gold_adversarial.json` with `implied_raws`; the offline test
  `test_every_trap_is_hard_caught` proves your band actually rejects it.

---

## 4. Honesty rules for plugin authors

The project's contract with the reader (and with itself) is that **no number
in a gold set, a figure or a README table is hand-typed**. Concretely:

1. Gold `expected` values are computed by the same calculator that will score
   them — generate the gold with the calc, don't write values by hand.
2. Every formula and constant has a `source` (a DOI, a standard equation, or —
   for software conventions like PLINK/ChAMP flags — the tool's own manual,
   labelled "software convention, not a literature value").
3. Every `sanity_band` is a real physical envelope, not a value tuned to pass
   the test set. The boundary evaluation exists precisely to catch bands that
   are too loose to be honest.

---

## 5. Example

Add a working example that ships with the repo (in `examples/` or as a pytest)
showing a `register_tool` call with a real pydantic model, a real calc, and a
Block. The existing `bioprinting` and `coculture` domains (added for the
LabMath-Bench) are deliberately written as self-contained templates:
`labwright/calc/bioprinting.py` + `labwright/calc/coculture.py` show a
calculator module with a documented source string, and `labwright/blocks.py`
shows their `Block` + bands. Copy either one.

---

## 6. Testing your plugin

Run the project checks after adding a domain:

```bash
python -m pytest                                   # 580 tests, all green
python -m eval.audit_claims                        # every README number re-derived
python paper/fig_tba.py results/eval_labmath_flash.json results/eval_labmath_pro.json
python paper/fig_ablation.py results/eval_labmath_flash.json results/eval_labmath_pro.json
python paper/fig_failsafe.py results/adversarial_flash.json results/adversarial_pro.json
```
