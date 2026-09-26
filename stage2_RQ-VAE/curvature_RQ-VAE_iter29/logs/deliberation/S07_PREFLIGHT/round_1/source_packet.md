# Iter29 S07_PREFLIGHT Source Packet — Round 1

STAGE_ID=S07_PREFLIGHT
ROUND=1

## Objective

Independently audit the applied iter29 implementation against the accepted source-of-truth, hypothesis, provenance, mechanism contract, one-factor record, S05 policy amendment, and Judge-approved S06 plan. Identify static/contract violations or confirm readiness for the official preflight. S07 is source verification, not a run decision based on Stage2 metrics.

## Canonical authority

- Repository `/home/wlia0047/ar57/wenyu/GeneRec/CLAUDE.md` and `skill://curvature-rqvae-iter` govern the iteration.
- Canonical iter29 S00–S05 records are in `logs/source_snapshot_iter29.md`, `logs/protocol_manifest_iter29.md`, `logs/hypothesis_iter29.md`, `logs/mechanism_manifest_iter29.md`, `logs/mechanism_contract_iter29.json`, and `logs/one_factor_diff_iter29.md`, with their stage Judges.
- S06 round 2 is `ACCEPT_A`; canonical implementation plan `logs/implementation_plan_iter29.md`, SHA-256 `eef646457f01d0c531c74d4cd97536c1dffd1e2abb02a00a824d29c365306486`; Judge `logs/deliberation/S06_IMPLEMENTATION/round_2/judge.md`.
- Round-2 S05 Judge `MERGE_AB` authorizes the SID-only reporting migration as `POLICY_HYGIENE_DIFFERENCES`, not a scientific factor.
- Parent/direct control is iter26 at commit `612a5a41dfe524205b6afa46370ad0dce377882`; fixed control curvature `[0.6145357379232853,0.5333020920777128,0.3814078098431606]`.
- The sole scientific change is FCCR-1 fixed mapping with unchanged vectors `branching=[19.324911558712664,1.4605688962651735,1.0148104414712726]`, `raw_residual_medians=[1.0,0.10941,0.09331]`; expected curvature `[1.3660953164241916,0.7347829661951981,0.6439958072706683]`.
- Physical JSON keys are `branching` and `raw_residual_medians`; `behavior_branching`/`raw_residual_median` are semantic S04 names only. Missing plural raw key must fail closed. S03 grants historical method/value provenance only, not checkpoint replay or historic byte identity.

## Files to inspect

Inspect the current iter29 primary source, not candidate plans as implementation evidence:

- `curvature_RQ-VAE.py`
- `curvature_config.py`
- `modules/rqvae.py`, `modules/quantize.py`, `modules/sid_quality.py`, and relevant `modules/step_checks.py`
- `scripts/compute_closed_form_curvature.py`, `scripts/computed_behavior_branching.json`, `scripts/mvg_check.py`, `scripts/grad_check.py`, `scripts/export_sids_for_stage3.py`, `scripts/run_stage3_iter29.py`
- `logs/mechanism_contract_iter29.json`, canonical S02/S03/S05 records, S06 plan/Judge

## Audit rubric

Each candidate independently reports evidence-based PASS/FAIL for:

1. **Formula/data**: single shared formula implementation; exact FCCR-1 constants/order; physical input keys; explicit raw-median fail-closed path; stored `x/y/u/c` derived values agree with inputs; no legacy alias, normalization, fallback, re-calibration, or old writer use; S03 limitations and separated source provenance remain intact.
2. **Contract consumer**: runtime recomputation compares input-derived intermediates and curvature to JSON and the exact unchanged ten-field S04 contract with strict finite/shape/domain checks and tolerance; fixed curvature is supplied to non-trainable buffers; no active cyclic schedule, trainable curvature, or curvature regularization.
3. **Stage2 behavior/policy**: Stage0/Stage1/warm-start, architecture, losses, optimizer, Sinkhorn rule/settings, seed/steps, reporting cadence, SID writes, final export, and warning-only metric handling remain locked; SID diagnostics are SID-derived only; no HR@50/subprocess/positional CLI or descriptive metric gate; `should_early_stop` is unconditional `(False, "")`.
4. **Routing/one-factor**: full mechanism name remains `iter29_bounded_rational_additive_mapping`; Stage2 products route only under `results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/`; Stage3 wrapper reads the iter29 Stage2 SID JSON and routes to short `results/stage3_T5Train/curvature_RQ-VAE_iter29/`; no output contamination of another iteration.
5. **MVG readiness (design audit only)**: existing helper is prepared for the later single S08 execution using one locked iter8 checkpoint and one real batch; verifies explicit `loss.backward()`, finite/nonzero model gradients, no curvature parameter/optimizer membership, fixed-buffer and mode/step/update invariance, zero curvature regularization, and same-state/same-batch candidate-vs-iter26 measurable curvature-consuming counterfactual. S07 candidates must not run it.
6. **Scope/minimality**: no unauthorized Stage1/Stage3 trainer, other iteration, mechanism, or protocol change; only S05-approved local policy cleanup/deletions are present.

For each finding cite exact source paths and line locations or a directly inspected value. Separate observed fact from interpretation. A test/preflight PASS must not be asserted unless observed by the orchestrator after Judge authorization.

## Boundaries

- A/B workers write only `logs/deliberation/S07_PREFLIGHT/round_1/agent_a.md` or `agent_b.md`, beginning with the exact independence header required by the skill.
- Do not edit production source or canonical artifacts. Do not run `preflight_contract.py`, `deliberation_gate.py`, MVG, pytest, builds, formatter, GPU work, Stage2, or Stage3. Judge C adjudicates the two audits and writes `judge.md`; only after an accepted S07 canonical decision does the orchestrator run the official no-argument `preflight_contract.py` and capture `logs/preflight_contract_iter29.log`.
- Do not run the global deliberation gate before its pre-Stage2 phase and required S08/S09 artifacts exist. No Stage2 or Stage3 launch is authorized by S07.
