# Contributing to Labwright

Labwright is designed to be extended by adding **calculators** — domain math
that the agent can call and the verifier can check. Everything else (agent,
verifier, demo, CLI) reads the same tool registry, so a new calculator is
instantly usable everywhere.

## Add a calculator (the whole story)

1. **Write the math** as a pure function in `labwright/calc/` (or a new
   module). Follow the existing style: docstring with the equation, units in
   parameter names, `_validate_positive` guards, no I/O side effects.

2. **Declare the tool** in `labwright/tools.py`:

   ```python
   from pydantic import BaseModel, Field

   class MyParams(BaseModel):
       x: float = Field(gt=0, description="What x is, in what unit")

   register_tool(MyParams, "my_calculator", "One-line description for the LLM", my_calc, "my_domain")
   ```

3. **Test it** in `tests/` against an independent analytic evaluation —
   not against the implementation. If you can't independently verify the
   number, don't add it: unverifiable calculators undermine the project's
   entire premise.

4. Run the suite:

   ```bash
   pytest tests/
   ```

## Rules

- **No fabricated numbers.** A calculator's test must reproduce a value you
  can derive by hand or cite. Literature constants need a source.
- **No derived-number generation in the LLM.** If a quantity is computable,
  the model must not be allowed to emit it directly.
- **Pure functions only** in `calc/`. Side effects (I/O, network) live in the
  agent/UI layer.
- Keep the registry discoverable: one `register_tool` call, one line in the
  README tool table if it's user-facing.

## Reporting issues

A bug report should include: the function called, inputs, actual output,
expected output, and — for literature-referenced constants — the source.
