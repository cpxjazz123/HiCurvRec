# Iter31 S07 Preflight — Frozen Source Packet

## Identity and authorization

- Stage: `S07_PREFLIGHT`, round 1; Iter31 HRA-STEP6-1, canonical parent Iter29.
- S06 Judge: `logs/deliberation/S06_IMPLEMENTATION/round_1/judge.md`, verdict `MERGE_AB`; canonical plan `logs/implementation_plan_iter31.md` is approved and has been applied once.
- Candidate tasks: independently audit the applied code/contracts/wiring and run the two authorized static preflights. No source edits, checker edits, MVG, Stage2/Stage3, GPU use, or user input.
- S07 checker execution is authorized; this does not authorize S08 MVG or any training.

## Canonical registration

Read, but do not modify, canonical S00–S06 records/contracts:

- `logs/source_snapshot_iter31.md`
- `logs/protocol_manifest_iter31.md`
- `logs/hypothesis_iter31.md`
- `logs/mechanism_manifest_iter31.md`
- `logs/mechanism_contract_iter31.json` — exact ten-field FCCR-1 fixed-curvature contract
- `logs/hra_step6_contract_iter31.json` — separate HRA-STEP6-1 contract
- `logs/one_factor_diff_iter31.md`
- `logs/implementation_plan_iter31.md`
- all `logs/deliberation/S00_SOURCE_TRUTH/round_1/judge.md` through `S06_IMPLEMENTATION/round_1/judge.md`

The only conceptual/model change is `RqVae._step6_sum_embeddings`. HRA expression must preserve tangent `e_l`, source-curvature `exp0(c_l)` then `log0(c_l)`, target-curvature `exp0(c0)`, right-nested L0/L1/L2 Möbius additions at `c0`, final `log0(c0)`, with `(L,D,B)` embeddings transposed to `(B,D)` before geometry helpers. The FCCR vector remains `[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]`; FCCR loader requires both `branching` and `raw_residual_medians`, recomputes it, and keeps the exact ten-field FCCR schema. Do not claim paper-equivalent telescope, runtime activation, or performance.

## Applied implementation inventory to audit

Selective Iter29 source copy under `stage2_RQ-VAE/curvature_RQ-VAE_iter31/`:

- Root: `curvature_RQ-VAE.py`, `curvature_config.py`.
- Modules: `modules/rqvae.py` (only `_step6_sum_embeddings` is a conceptual delta); unchanged `modules/{sid_quality.py,quantize.py,__init__.py,encoder.py,hab.py,hyperbolic.py,loss.py,normalize.py,step_checks.py,utils.py,tokenizer/semids.py}`.
- Data: unchanged `data/{__init__.py,amazon.py,instruments.py,ml1m.py,ml32m.py,preprocessing.py,processed.py,schemas.py,utils.py}`.
- Init: unchanged `init/kmeans.py`.
- Scripts: `scripts/compute_closed_form_curvature.py` (identity/docstring only), unchanged frozen `scripts/computed_behavior_branching.json`, unchanged `scripts/export_sids_for_stage3.py`, rewritten audit-only `scripts/mvg_check.py`, delegated `scripts/grad_check.py` with Iter31 identity, new `scripts/preflight_hra_step6_iter31.py`.
- Stage3 direct route: `stage3_T5Train/train_HG-Rec.py` has only hard-coded path/variant/launcher-log wiring changes; model/data/training/evaluator/metric/protocol settings stay unchanged. No Iter31 Stage3 wrapper exists.
- Do not find/copy/expect Iter29 result files in the Iter31 source tree. Only the historical Iter8 checkpoint remains an external warm start.

## Required static audit items

1. Read and compare current source to canonical S06 plan; confirm only the registered Step6 operation changes model behavior, no helper/Step4/Step5/forward/decoder/loss/quantizer/data/optimizer/Sinkhorn/schedule changes, no extra losses/d-HSTE, no CLI/environment parameter overrides.
2. Inspect the exact Step6 AST/source operation, order, shape and finiteness guards; confirm the only intended scientific difference from Iter29 is `_step6_sum_embeddings` and inherited `modules/hyperbolic.py`, `modules/quantize.py`, `modules/loss.py` remain unchanged.
3. Verify HRA and FCCR JSON files stay distinct; exact FCCR ten-field schema and vector; exact historical input key pair; Iter31 loader points to `logs/mechanism_contract_iter31.json` and retains exact validation.
4. Verify the new HRA checker is static/no-argument/no-env, reads both canonical contracts, validates source structure, and explicitly disclaims runtime proof. Do not edit it if a check fails; report the precise failure for Judge/orchestrator.
5. Verify fixed result paths: Stage2 `RQVAE_OUT_DIR` and every Stage2 product path resolve under `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/`; Stage2 source tree contains no `.pth`, `.pt`, `.npy`, `item_sids.json`, parent results/logs, caches or old wrapper.
6. Verify Stage3 route constants point exactly to Iter31 `item_sids.json`, variant `iter31_hra_step6_common_reference`, short `LOG_PATH`/`SAVE_PATH`, and internal `_stage3_launcher.log` under the short results root. Confirm `_LAUNCHER['script']` remains direct `__file__`. Outer log will later be redirected to `results/stage3_T5Train/curvature_RQ-VAE_iter31/logs/_stage3_run.log`; do not launch Stage3 at S07.
7. Do not treat Stage2 SID-quality measures as gates or infer Stage3 benefit from a static check.

## Authorized S07 commands

Each candidate, from cwd `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter31`, may separately run these no-argument scripts and report their full outputs:

1. `python /home/wlia0047/.claude/skills/curvature-rqvae-iter/scripts/preflight_contract.py` — unchanged shared FCCR-1 checker, no modifications/bypass.
2. `python scripts/preflight_hra_step6_iter31.py` — distinct Iter31 HRA checker, no arguments/environment overrides.

Both must pass independently. A pass confirms only static contract/source checks, not runtime HRA activation or gradients. No checker runs are authorized before this S07 stage or outside the two separate invocations above. No tests/build/formatter/project suite.

## S08 and later remain blocked on S07 adjudication

After Judge accepts S07, S08 independently adjudicates one same-checkpoint/one-batch MVG: compare actual HRA Step6 and legacy Euclidean output on identical quantized embeddings/state/fixed curvature; measure helper projection/log clamp, Möbius/domain/finiteness, fixed-curvature/optimizer exclusion, and `loss.backward()` plus relevant component gradients. No numeric direct-effect threshold is registered. A nonfinite/out-of-ball/gradient failure, or an inactive mechanism that requires changing its locked equation to activate, blocks Iter31 under the skill; no tuning/sweeps.

Stage2 requires S09 approval and root `CLAUDE.md` §6 checks; Stage3 requires S11 route/evaluation approval and one run. These are not authorized by S07.

Each candidate returns `ROLE=AGENT_A` or `ROLE=AGENT_B`, independent audit conclusion, exact source locations, both separate checker outputs/commands and interpretation, all deviations/blockers, no-edit/no-training statement, `USER_INPUT_REQUIRED=NO`. Judge C reviews candidate evidence and writes canonical `logs/deliberation/S07_PREFLIGHT/round_1/judge.md` and `logs/preflight_iter31.md`; no source edits or training execution.