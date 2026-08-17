# Supplementary traceability log

Every JSON in this directory is one benchmark entry × one design-path system (labwright / labwright_iter / tool_no_gate / finetuned), rebuilt from the committed `results/eval_labmath_*.json` files by `python -m eval.make_traceability_log`. Each file carries the full DesignPlan JSON the agent produced, its computation provenance (one record per derived field: formula, inputs with units, value, unit, verifier status, code version) and the ordered tool-call trace.

Coverage: **1132** entries with a plan + provenance; **39** distinct derived fields; 2 model file(s).

Most-used tools across the traced entries:
    submit_design 1328 · bioprinting_extrusion_volume_nl 249 · enzyme_fractional_activity 228 · coculture_cells_per_well 217 · coculture_seeding_ratio 216 · residence_time 215 · reynolds_number 181 · bioprinting_print_time 171

`INDEX.json` holds the full aggregate. The number on every edge of `paper/fig_protocol_dag.py` and every value in the LabMath-Bench tables can be traced to a record here.
