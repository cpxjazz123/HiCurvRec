ROLE=AGENT_A
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=logs/deliberation/S07_PREFLIGHT/round_2/source_packet.md
STAGE_ID=S07_PREFLIGHT
ROUND=2

# Agent A — narrow runtime identity correction

## Evidence and scope

The round-1 Judge confirms the active runtime consumer recomputes FCCR-1 intermediates and curvature from the JSON's physical inputs and compares derived vectors/final curvature, but does not enforce the approved input values/order. The ten-field S04 contract records semantic `formula_inputs` and final curvature, not input vectors; final-curvature agreement therefore cannot establish the registered provenance identity. Round-1 `REPLAN_CONSTRAINTS` bind this correction to exact ordered input checks at the iter29 runtime consumer, retaining physical JSON keys `branching` and `raw_residual_medians`, the canonical S04 schema, and the registered mapping/protocol.

S03 approves historical method/value provenance only (not replay or historical byte identity), with `[L0,L1,L2]` vectors in `logs/mechanism_manifest_iter29.md:22-27`. The existing JSON uses those physical keys and values (`scripts/computed_behavior_branching.json:7-16`). The existing loader directly reads both keys and computes via the shared helper (`curvature_RQ-VAE.py:290-302`); `_compare_formula_vector` currently permits a `1e-9` absolute tolerance for derived-data comparisons (`curvature_RQ-VAE.py:258-277`). The formula in `scripts/compute_closed_form_curvature.py:11-56` transforms each layer independently as `x=B/(B+2)`, `y=m/(m+0.1)`, `u=(x+y)/2`, `c=0.05+1.45u`. S06 identifies the inputs, order, constants, and formula as locked (`logs/implementation_plan_iter29.md:15-35,53-59`).

## Minimal source correction proposal (not applied)

Files/symbols:

1. `scripts/compute_closed_form_curvature.py`: add two immutable registered-input constants beside the existing FCCR-1 constants:
   - `REGISTERED_BRANCHING = (19.324911558712664, 1.4605688962651735, 1.0148104414712726)`
   - `REGISTERED_RAW_RESIDUAL_MEDIANS = (1.0, 0.10941, 0.09331)`
2. `curvature_RQ-VAE.py`: in `_load_closed_form_curvatures()`, immediately after confirming the JSON object and the presence of both explicit physical keys, validate each raw input vector against its corresponding registered constant **before** calling `compute_closed_form_curvature()`.

Use a small local validation helper or inline loop at the loader boundary. For each vector, require a JSON `list` of exactly `N_LAYERS` elements; reject booleans and all non-`int`/`float` elements; convert accepted numeric elements to `float`, reject any non-finite value, and compare the resulting ordered three-value list to the registered tuple using exact numeric equality (zero tolerance, no `allclose`). Raise `ValueError` identifying the physical key and expected/observed values on any mismatch. This rejects missing, malformed, reordered, or merely different valid inputs closed before the formula output is returned to model construction/training.

Keep the physical key names unchanged: `branching` is the JSON key for semantic `behavior_branching`; `raw_residual_medians` is the JSON key for semantic `raw_residual_median`. Do not rename/alias either key. Put the identity assertion in the active iter29 runtime consumer, not in the generic shared `compute_closed_form_curvature()` formula API: the helper remains usable for arbitrary valid vectors and remains the single formula implementation. The two registered constants colocated with its FCCR-1 constants avoid a second independently maintained literal set in the loader; no schema or API change is needed.

For the canonical vectors, the existing helper call and every mapping constant, derived `x_l/y_l/u_l`, `closed_form_c_l`, contract field/value, and fixed-buffer path remain unchanged. Preserve the existing `1e-9` formula/data/contract comparisons for recomputed intermediates and outputs; those checks answer a different question from exact registered-input identity and must not substitute for it. No tolerance is allowed on the input identity check. Historical provenance remains explicitly limited as S03 states. This does not change warm-start/export behavior or any training behavior.

## Focused bounded acceptance scenario (proposal only; not run)

- Canonical `branching=[19.324911558712664,1.4605688962651735,1.0148104414712726]` and `raw_residual_medians=[1.0,0.10941,0.09331]` pass the new identity check and continue to produce the current `x/y/u/c` and contract values.
- Construct a valid distinct compensated pair at L0 while leaving L1/L2 unchanged. Let
  `x0 = 19.324911558712664 / (19.324911558712664 + 2.0)`,
  `y0 = 1.0 / (1.0 + 0.1)`,
  `B0' = 18.0`,
  `y0' = x0 + y0 - 18.0/(18.0+2.0)`, and
  `m0' = 0.1*y0'/(1-y0')`.
  Here `x0≈0.906213`, `y0=10/11≈0.909091`, so `y0'≈0.915304` lies strictly between 0 and 1 and `m0'` is finite and positive (approximately 1.095). Thus the changed pair obeys the helper's valid domains (`B0'≥0`, `m0'>0`) and is distinct from the approved L0 pair. By construction `x0'+y0'=x0+y0`, so `u0' = u0` and the affine curvature formula gives exactly the same mathematical `c0`; L1/L2 are unchanged, hence the entire final curvature vector is unchanged. The runtime must nevertheless raise `ValueError` on the first noncanonical physical input vector, before accepting the recomputed/final vector or reaching Stage2. This directly demonstrates why output equality alone is insufficient.

This scenario is an acceptance specification only. No source changes, gates, tests, MVG, or training were performed for this proposal.