# Iter31 S04 Contract — frozen source packet

```text
STAGE_ID=S04_CONTRACT
ROUND=1
ITERATION=31
ITERATION_PURPOSE=PERFORMANCE_SEEKING_MECHANISM
SWEEP_OR_REPLICATION_ITERATION=NO
ROOT_CAUSE_ITERATION=NO
```

## Canonical inputs

Use only the following judge-approved artifacts as active instructions:

- `logs/source_snapshot_iter31.md` — S00 `MERGE_AB`
- `logs/protocol_manifest_iter31.md` — S01 `MERGE_AB`
- `logs/hypothesis_iter31.md` — S02 `MERGE_AB`
- `logs/mechanism_manifest_iter31.md` — S03 `MERGE_AB`, pass limited to historical method/value provenance with replay/hash limits
- their corresponding S00–S03 Judge reports

Do not treat rejected candidate drafts or historical summaries as active direction. Verify all checker/source claims directly against primary files.

## Decision to make

S02 registered one performance-seeking change: replace only Iter29's Euclidean Step6 tangent embedding sum with the exact HRA-inspired common-reference aggregation below. S02 deliberately left FCCR-1 active and authorized no contract change. S03 passed with explicit historic replay limits but did not authorize a contract transition. S04 must now independently adjudicate whether the user-requested Iter31 direction can be activated through a valid between-iteration transition from cancelled Iter30, and define the complete machine-readable contract and preflight path without bypassing or silently altering existing gates. `USER_INPUT_REQUIRED=NO`; use evidence and autonomous adjudication.

If the HRA change cannot be represented and checked while preserving Iter29's fixed-curvature source contract and the exact S02 one-factor scope, reject/abort according to skill rules. Do not change HRA equation, values, parent, seed, or Stage3 protocol to make a contract fit. No code/checker edit, implementation, or training is authorized by S04.

## S02 equation — immutable candidate specification

For `l ∈ {0,1,2}`, `e_l` is the selected quantizer embedding in tangent coordinates (with the existing STE forward value during training), not a ball point or token ID. Preserve exact L0,L1,L2 order:

```text
e_l   = tangent-coordinate quantizer embedding
q_l   = exp0^{c_l}(e_l) ∈ B_{c_l}
q_l^0 = exp0^{c0}(log0^{c_l}(q_l)) ∈ B_{c0}
h     = q_0^0 ⊕_{c0} (q_1^0 ⊕_{c0} q_2^0)
z     = log0^{c0}(h) ∈ T_0(B_{c0})
c0    = c_0 = 1.3660953164241916
```

`z` replaces only the Euclidean Step6 tangent sum and is consumed by the existing decoder; decoder, reconstruction loss, quantizer, residual Step4/Step5, codebooks/assignment, all losses, optimizer, Stage1 and Stage3 remain unchanged. Do not reassociate or reorder the non-associative Möbius operation. Do not apply `log0` directly to `e_l`. The unclipped pointwise exp/log identity is conditional; helpers project exp outputs and clamp log inputs. No cross-curvature homomorphism, exact telescope, residual inversion, or performance result is claimed. d-HSTE is excluded.

## Inherited FCCR-1 fixed-curvature substrate

The exact inherited layer-order vector is `[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]`. Historical inputs are exactly `branching` and explicit `raw_residual_medians` in the Iter29 JSON, with documented but unreplayed historical measurement provenance. Do not substitute normalized `[0.001,0.932889,1.0]`, an ambiguous `residual_norm`, or a newly refreshed measurement. Curvature is fixed/nontrainable/non-time-varying; preserve current fixed-buffer behavior and unchanged residual geometry.

## Primary preflight and runtime contract facts to verify

1. `.claude/skills/curvature-rqvae-iter/scripts/preflight_contract.py` is FCCR-1-specific. It requires `protocol_manifest_iter<N>.md`, `hypothesis_iter<N>.md`, `mechanism_manifest_iter<N>.md`, `mechanism_contract_iter<N>.json`, and `one_factor_diff_iter<N>.md`; enforces `contract_version=FCCR-1`, closed-form fixed curvature, exact formula inputs `['behavior_branching','raw_residual_median']`, three finite positive final values; and statically rejects trainable curvature, step-dependent `get_c()`, missing fixed buffer, and nonzero curvature regularization. It prints `MECHANISM_CONTRACT_PASS`. It contains no HRA Step6 check.
2. Iter29 `curvature_RQ-VAE.py::_load_closed_form_curvatures` reads `logs/mechanism_contract_iter29.json` and requires the exact ten-field FCCR-1 schema, while requiring explicit `raw_residual_medians` and recomputing/validating the fixed curvature. The iteration copy must preserve a compatible curvature contract path; adding unrelated keys to that exact JSON would make the loader fail unless a separately approved code design changes it.
3. `.claude/skills/curvature-rqvae-iter/scripts/deliberation_gate.py` requires `logs/mechanism_contract_iter<N>.json` as S04's canonical file before Stage2. Additional S04 canonical artifacts may be declared if justified; the Judge must list each.
4. S03 explicitly leaves S04/S07 to resolve compatibility by adjudication; it does not authorize bypassing or modifying a checker.
5. Root `CLAUDE.md` Stage2 artifact paths, no-CLI/no-env-override rules, no-gate policy, and pre-training §6 gradient-path check remain binding. S04 does not authorize any training.

Candidates must examine whether the contract can cleanly preserve the exact FCCR-1 fixed-curvature sub-contract while separately expressing and checking the HRA-only Step6 transition, or whether another source-supported route is required. No path may leave the HRA operation unrepresented/unverified, make the FCCR-1 checker falsely pass as the only HRA check, silently disable a checker, add a second mechanism, or alter the scientific equation.

## Required candidate deliverable

Each candidate independently returns a proposed verdict and machine contract specification, including:

- whether a valid between-iteration contract transition can activate S02 HRA for Iter31;
- exact contract version(s), schema/file paths, immutable values/equation, inherited FCCR-1 fields, and unchanged factors;
- the preflight/checker execution path that will prove both the FCCR-1 curvature invariants and HRA Step6 invariants, including which later stage authorizes any checker implementation/update;
- how every HRA semantic/domain requirement, no-telescope limit, raw-data provenance limitation, and strict one-factor scope are represented;
- any infeasibility/hard gap, downstream constraints, self-rejection criteria, confidence, `USER_INPUT_REQUIRED=NO`;
- explicit statement that S04 authorizes no source code, checker edits, Stage2/Stage3 run, training, or performance claim.

Use primary source evidence, distinguish direct facts from proposal/inference, and never invent a numerical performance threshold or new mechanism. Judge C must verify the candidates independently and write the canonical S04 contract JSON plus `judge.md`; if multiple canonical contract files are selected, list all. The skill's mandatory `mechanism_contract_iter31.json` must remain present. Only Judge-approved canonical contracts may propagate to S05.
