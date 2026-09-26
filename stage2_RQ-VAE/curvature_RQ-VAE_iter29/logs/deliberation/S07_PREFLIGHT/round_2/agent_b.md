ROLE=AGENT_B
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=logs/deliberation/S07_PREFLIGHT/round_2/source_packet.md
STAGE_ID=S07_PREFLIGHT
ROUND=2

## Finding and evidence

The active consumer is `curvature_RQ-VAE.py::_load_closed_form_curvatures()` (lines 280–350). It requires the physical JSON keys `branching` and `raw_residual_medians`, calls `compute_closed_form_curvature()` on those inputs, and checks the stored `x_l`, `y_l`, `u_l`, JSON curvature, and contract curvature against recomputed outputs. It does not compare either physical input vector to the registered S02/S03 values. The S04 contract (`logs/mechanism_contract_iter29.json`, fields 1–12) has semantic `formula_inputs` and final curvature, but no physical input vectors. Therefore matching final curvature does not enforce approved input identity.

The approved physical ordered vectors are `branching` (semantic name `behavior_branching`) = `[19.324911558712664, 1.4605688962651735, 1.0148104414712726]` and `raw_residual_medians` (semantic formula name `raw_residual_median`) = `[1.0, 0.10941, 0.09331]` (`logs/mechanism_manifest_iter29.md:22–27`; source packet lines 22–26). Keep these physical JSON keys distinct from the semantic names in the unchanged contract. The shared formula helper validates vector shape/types/domains and computes the registered map (`scripts/compute_closed_form_curvature.py:18–62`), but cannot establish that a valid vector is the approved vector.

## Minimal correction proposal (not implemented)

Change only the runtime input-validation path in `curvature_RQ-VAE.py::_load_closed_form_curvatures()`. Define the two registered expected vectors once as explicit iter29 constants adjacent to the loader (or at its existing formula constants), then, after the existing object/key-presence checks and **before** invoking `compute_closed_form_curvature()`, call the existing `_compare_formula_vector(payload, physical_key, expected_vector)` for each physical key:

- `_compare_formula_vector(payload, "branching", APPROVED_BEHAVIOR_BRANCHING)`
- `_compare_formula_vector(payload, "raw_residual_medians", APPROVED_RAW_RESIDUAL_MEDIANS)`

Populate the expected vectors with the exact ordered values above. This reuses existing fail-closed handling: require a JSON list of exactly `N_LAYERS` elements; reject booleans and non-numeric values; reject non-finite values; compare in-place order with `rtol=0.0` and `atol=FORMULA_COMPARISON_TOL` (`1e-9`). Missing keys already fail before the calls; do not coerce, sort, reorder, broadcast, or fall back to aliases. A mismatch raises `ValueError` during loader execution, before the resulting curvature can initialize training. Keep the ten-field S04 contract, formula/helper/API, mapping constants, derived arrays, final curvature, provenance, and all training behavior unchanged. Do not also put this identity policy into the shared formula helper: it is the active iter29 runtime consumer that must fail closed, and a second copy there would duplicate the approved-vector policy without strengthening the training boundary. No official preflight or other gate needs modification for this bounded source correction.

For canonical inputs these two additional comparisons pass, after which existing formula and stored-output checks run unchanged. This preserves the original formula/API and the outputs `[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]` within the already declared tolerance.

## Focused bounded acceptance scenario

Check the canonical vectors pass the two input checks. Then construct a distinct valid compensated pair changing only L0. Let the canonical transformed L0 values be `x0 = 19.324911558712664 / (19.324911558712664 + 2)` and `y0 = 1 / (1 + 0.1)`. Set `B0' = 18`, so `x0' = 18 / 20 = 0.9`; set `y0' = x0 + y0 - 0.9`, and `m0' = 0.1 * y0' / (1 - y0')`. Use `branching' = [18, 1.4605688962651735, 1.0148104414712726]` and `raw_residual_medians' = [m0', 0.10941, 0.09331]`. Since `0 < x0,y0,x0',y0' < 1`, these are finite, in-domain inputs (`B' >= 0`, `m' > 0`); by construction `x0' + y0' = x0 + y0`, so `u0' = u0` and `c0' = c0` algebraically. L1/L2 are unchanged, thus the complete final-curvature vector is unchanged, despite both physical vectors being noncanonical. The input pin must reject the changed `branching` vector before formula output is consumed (and would also reject the changed residual vector if checked independently). This exercises the precise failure mode that final-curvature equality alone cannot detect.

## Scope and status

This is a proposal only. It does not change source, S04, provenance claims, mapping, parent/control, protocol, or any later S09 launch behavior. No tests, preflight/deliberation gates, MVG, build/formatter, GPU work, Stage2, or Stage3 were run; no validation result is claimed.