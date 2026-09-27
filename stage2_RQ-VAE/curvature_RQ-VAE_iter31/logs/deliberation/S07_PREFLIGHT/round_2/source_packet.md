# Iter31 S07 Preflight — Round 2 Repair Verification

## Identity and governing adjudication

- Stage: `S07_PREFLIGHT`, round 2; Iter31 HRA-STEP6-1, parent Iter29.
- Round 1 Judge: `logs/deliberation/S07_PREFLIGHT/round_1/judge.md`, `VERDICT=REJECT_BOTH`; canonical outcomes in `logs/preflight_iter31.md`.
- This round verifies only the two S07 Judge-authorized repairs. It is not S08 and authorizes no MVG, GPU, Stage2, or Stage3.
- Canonical S00-S06 decisions/contracts remain unchanged; FCCR-1 shared checker remains unchanged and cannot be bypassed.

## Exact authorized repairs now applied

1. `logs/mechanism_manifest_iter31.md`: appended one additive semantic-vocabulary subsection containing the exact required phrases **behavior branching**, **raw residual**, and **normalized layer scale**. It restates existing S03 facts only: `branching` input/provenance; raw residual median exact key `raw_residual_medians`; normalized scale is separate and not a substitute. All pre-existing manifest content/values/provenance limits/formula/schema remain unchanged.
2. `scripts/preflight_hra_step6_iter31.py::_find_method`: the comprehension now appends the matching `child` FunctionDef/AsyncFunctionDef rather than the containing `RqVae` ClassDef; exact-one-match check and return remain unchanged.

Round 1 Judge authorized no other edits. Inspect exact diff/content and report any unauthorized change as a failure; candidates do not edit/fix.

## Canonical implementation and checker scope

Read the frozen round-1 S07 packet, round-1 Judge/outcome, S06 plan/Judge, canonical S00-S06, actual Iter31 source and both checkers. Binding paths:

- `logs/mechanism_contract_iter31.json`: exact separate ten-field FCCR-1 contract and fixed vector `[1.3660953164241916,0.7347829661951981,0.6439958072706683]`.
- `logs/hra_step6_contract_iter31.json`: distinct HRA-STEP6-1 equation/transition/scope/domain contract.
- `curvature_RQ-VAE.py`: exact FCCR loader path `logs/mechanism_contract_iter31.json`, explicit `branching` + `raw_residual_medians`, formula recomputation and unchanged exact schema.
- `modules/rqvae.py::RqVae._step6_sum_embeddings`: one scientific change only; tangent `(L,D,B)`→`(B,D)`, source-curvature exp/log, target `c0` exp, right-nested L0/L1/L2 Möbius operations, final c0 log, decoder shape/finiteness.
- `scripts/preflight_hra_step6_iter31.py`: distinct AST/contract audit. It must say runtime activation is not established.
- Stage2 product paths remain under `results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/`; no output product under source tree. Stage3 path wiring remains statically routed, still S11-only.

No model/HRA formula, fixed values, contract/schema, checker policy, data, optimizer/Sinkhorn, Stage3 model/evaluator/protocol, or training settings change is allowed in this round.

## Parent-side checker commands and execution procedure

Candidate tasks perform independent source audits; candidates must not edit files or attempt command substitutes. Parent Main will run each checker **once** for this S07 round, separately, and provide exact stdout/stderr/exit status to Judge C. Use the same cwd for both:

`/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter31`

1. Unchanged shared FCCR checker (absolute path is required; the two round-1 wrong-path attempts are documented and must not be repeated):

`python /home/wlia0047/ar57/wenyu/GeneRec/.claude/skills/curvature-rqvae-iter/scripts/preflight_contract.py`

2. Distinct Iter31 HRA checker:

`python scripts/preflight_hra_step6_iter31.py`

Each script has no experiment arguments. Run them separately. Both must exit 0 and report their own PASS markers. HRA pass is static-only; it is not S08 activation, geometry-domain, gradient, or performance proof. Do not run other tests/builds/checkers/MVG/training/GPU.

## Independent candidate requirements

Agent A and Agent B each independently inspect the two authorized changes against round-1 Judge scope, verify unchanged canonical contracts and source boundary, inspect current code and parent comparison, check source-tree product exclusion/Stage2 path and Stage3 wiring, and return evidence with file/symbol/line locations. They do not run commands or edit files. Explicitly state that checker execution outcomes are parent-side and will be adjudicated from raw tool results.

Judge C reads both full reports plus parent raw command results and primary source; writes canonical `logs/deliberation/S07_PREFLIGHT/round_2/judge.md` and updates `logs/preflight_iter31.md` with both actual command outputs/exit statuses and final verdict. Only if both checks pass and no other boundary violation exists may S07 PASS and allow progression to the separately deliberated S08. S08, Stage2, Stage3 and GPU remain unauthorized until their own canonical gates.