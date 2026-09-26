# Iter29 S12 source packet — Stage3 outcome and autonomous next direction

ROLE=SOURCE_PACKET
INDEPENDENCE_DECLARATION=Shared primary-evidence packet; Agent A and Agent B independently classify it and must not read each other's candidate.
STAGE_ID=S12_RESULT_CLASSIFICATION
ROUND=1
ITER=29

## Objective and hard boundaries

Classify iter29's completed, once-only Stage3 result from direct primary evidence. Separate mechanism status from promotion status. Decide whether the result is protocol-valid, whether the observed candidate/direct-control delta supports positive/neutral/negative attribution given the known noise limitation, whether the inclusive user target was reached, whether S14 is newly triggered, and the next evidence-supported autonomous action. The task threshold is `test_recall@10 >= 0.065`. Do not rank against iter18: S01 marks it `HISTORICAL_NONCOMPARABLE`; iter26 is the sole direct control. Do not ask for user input or propose a retry/restart of iter29.

## Registered experiment and prior direction

- S01 canonical: `logs/protocol_manifest_iter29.md`; protocol ID `FCCR-1_iter29_bounded_alternative_closed_form_vs_iter26`; single factor; parent/direct control/canonical baseline iter26; seed 42; Stage2 100,000 steps; Stage3 seed 42, 150 epochs, beam 20, expected `n_eval=57439`. Iter18 is historical-noncomparable. The user threshold is inclusive `>=0.065`.
- S02 `logs/hypothesis_iter29.md`: sole mechanism delta is the bounded rational/additive fixed-curvature mapping using behavior branching and historical raw residual medians: `x=B/(B+2)`, `y=m_raw/(m_raw+0.1)`, `u=(x+y)/2`, `c=0.05+1.45*u`; registered `c=[1.3660953164241916,0.7347829661951981,0.6439958072706683]`. Raw-residual values retain S03's historical-provenance/replay limitation.
- S03 `logs/mechanism_manifest_iter29.md` and `logs/deliberation/S03_PROVENANCE/round_1/judge.md`: PASS for historical method/value provenance only, medium confidence for raw residuals; no checkpoint-level reproduction or historical byte identity claim.
- S08 `logs/mvg_check_iter29.log`: MVG PASS; fixed-c invariance and counterfactual activation shown. Candidate quantize loss 2.10162615776062 versus control 0.8013777732849121; assignment fraction changed 0.4062500298023224. This proves implementation activation, not recommendation efficacy.
- S10 `logs/sid_geometry_iter29.md`, `logs/stage2_output_integrity_iter29.log`: Stage2 one run, four ranks, exit 0, 100,000 steps; fixed-c buffers/invariance; aligned JSON/4-token SID export and `STAGE2_OUTPUT_INTEGRITY_PASS`. Gini/collision/entropy are descriptive only; never use them as gates.
- Latest prior Global Review: `stage2_RQ-VAE/curvature_RQ-VAE_iter28/logs/global_review_after_iter28.md`. It says the formal three-clean-protocol-valid-iteration trigger was not met at that review, iter27 was aborted, iter25/28 were not clean FCCR-1 tests, iter26 was the only clean fixed-curvature test then, and it recommends one bounded alternative fixed mapping under FCCR-1. It reports an approximate historical noise band around 0.003 that was not independently estimated. Assess whether the current result triggers a new review; do not assume based only on iteration numbers.

## One supervised Stage3 run — primary evidence

- Approved method/one-run authorization: `logs/stage3_evaluation_plan_iter29.md`; S11 round 1 Judge `logs/deliberation/S11_STAGE3_EVALUATION/round_1/judge.md`; exact JSON-hash documentary correction adjudicated in round 2 `logs/deliberation/S11_STAGE3_EVALUATION/round_2/judge.md` and materialized at plan line 14. The prelaunch evidence is `logs/stage3_preflight_iter29.log`; post-comment-correction no-GPU/no-result-path smoke is `logs/stage3_launcher_smoke_iter29.log`.
- Exactly one Hub supervisor: `iter29-stage3`, PID 43614, terminal exit 0, uptime 37m37s, restarts 0. Command was the approved no-argument wrapper under the iter29 source directory; outer log path and both result roots stayed under `results/stage3_T5Train/curvature_RQ-VAE_iter29/`. No second launch/retry occurred.
- Exact timestamped run: `results/stage3_T5Train/curvature_RQ-VAE_iter29/logs/Amazon_2023_Instruments/Sep-27-2026_06-11-34/`; checkpoint directory is the matching short-root path under `ckpt/`.
- `test_final.json` primary content:
  - `n_eval=57439`
  - `test_recall@5=0.03953759640662268`
  - `test_recall@10=0.05921064085377531`
  - `test_ndcg@5=0.026252776900288842`
  - `test_ndcg@10=0.03257647953179143`
  - `best_checkpoint=/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter29/ckpt/Amazon_2023_Instruments/Sep-27-2026_06-11-34/HG_Rec_best.pth`
- Candidate `test_final.json` SHA-256 `b07de15ec53e8b423f3426f091a142b00260f084bd319abf6c7cb5d605583662`, size 351, mtime 2026-09-27 06:48:32 +1000 (epoch 1790455712). All four metrics were parsed and checked finite; one `test` event at `training_metrics.jsonl:278`, then `train_end` at line 279; test event epoch 150 and `n_eval=57439`.
- `training_metrics.jsonl` SHA-256 `412860c203e20498e17ffd9c7a87d5e39836fcee563f1835f503e1df65a1b88f`, size 87,793, same final mtime. It records `world_size=4`, `no_eval=true`, `num_epochs=150`, exact Stage2 SID `code_path`, and correct short-root log/ckpt paths.
- `HG_Rec.log` SHA-256 `2527330613702fff9e18abe23902432f217720473cf890207cb8bae7bfc0773c`, size 43,247, mtime after start; logged fixed training config and final test dictionary using the exact best checkpoint above.
- `_stage3_launcher.log` SHA-256 `fe9a49a8d7a4f46ffc0a91d20cf6a4768696bc54b778774714231f029f283806`, size 2,531,963, final mtime 2026-09-27 06:48:35 +1000. It shows torchrun/NCCL version and ranks 0–3, full training through epoch 150, and successful completion/test. A scan found no Traceback, RuntimeError, FloatingPointError, OOM, NCCL ERR or ERROR match.
- `HG_Rec_best.pth` SHA-256 `70facbde7de7961bbe71e23be6f4f951f30e696b959d04be66cdf7d32dab7e94`, size 18,283,796, mtime 2026-09-27 06:40:12 +1000; final test names this same checkpoint.
- Stage3 destination has exactly one timestamped run. `_ddp_sync/shutdown_signal.txt` contains `test_done` and was written at the final test time; it is a current completion marker, not a stale prior-attempt signal. No `early_stop_signal.txt` exists. `_stage3_run.log` exists at the approved path but is 0 bytes; torchrun stdout/stderr were captured in the nonempty launcher log.
- The effective `HG_Rec.log` configuration records `no_eval=True`, `skip_test=False`, empty `screen_baseline_log`, seed 42, batch 4096, 150 epochs, beam 20, TOPK `[5,10]`, exclude-history true, exact Stage0 root/Splits and exact Stage2 SID path. The final test event/file proves test was not skipped.
- Four per-rank `NCCL WARN lib wrapper not initialized` messages and a c10d barrier device-context UserWarning appeared at startup; training/test completed with exit 0. Preserve as operational warning evidence, not as a silent clean-run claim.

## Output identity and comparison

- Sole direct-control result: `results/stage3_T5Train/curvature_RQ-VAE_iter26/logs/Amazon_2023_Instruments/Sep-26-2026_18-11-20/test_final.json`; its exact values are `n_eval=57439`, R@5 `0.03788366789115409`, R@10 `0.057017009349048554`, NDCG@5 `0.025169911931406087`, NDCG@10 `0.03133359761524377`.
- Observed deltas (iter29 minus iter26, arithmetic independently evaluated): R@5 `+0.0016539285154685904`; R@10 `+0.002193631504726755`; NDCG@5 `+0.0010828649688827546`; NDCG@10 `+0.0012428819165476584`.
- Target shortfall: `0.065 - 0.05921064085377531 = 0.005789359146224693`; promotion target is not reached. Do not report an iter29-versus-iter18 direct delta or rank iter18; its protocol manifest is absent and S01 marks it historical-noncomparable.
- S14's approximate ~0.003 historical noise band was explicitly not independently estimated; iter29's `+0.0021936` R@10 delta is smaller. Treat the observed delta separately from any robust/causal improvement claim; a single run cannot estimate run-to-run variance.

## Metadata and execution discrepancies to assess

1. Wrapper source `scripts/run_stage3_iter29.py:19` sets `RQVAE_VARIANT="iter29_bounded_rational_additive_mapping"`; the final approved smoke asserts this. But the actual `training_metrics.jsonl` train_start event emits `variant="unknown_variant"`. `train_HG-Rec.py:843` writes `config.get("variant", "unknown_variant")`; source search found no other `variant` field consumer. The same event contains the exact correct `code_path`, short-root paths and `world_size=4`. Determine whether this is logging metadata only or invalidates any contract/provenance claim. Do not alter the completed run or reinterpret the result without evidence.
2. NCCL startup warnings listed above did not prevent training, test, or exit 0. Decide their classification relevance from completed primary logs.
3. S03's medium-confidence historical provenance/replay limit remains disclosed. Do not upgrade it to a checkpoint-level reproduction or silently call it invalid against the existing Judge decision.

## Independent candidate task

Agent A and Agent B must independently read this packet and primary files, then write only their role-specific candidate (`agent_a.md` or `agent_b.md`) under this round directory. Each candidate must begin with the required role, independence declaration, source packet path, and stage ID; state evidence, assumptions, risk, exact mechanism status from the allowed set, separate promotion status, validity/confounds, noise/causal limitations, whether S14 is triggered, and a concrete autonomous next direction/action. Use `USER_INPUT_REQUIRED=NO`. Do not edit canonical outcomes, source, metrics, or other shared files. Do not ask the user, retry Stage3, treat Stage2 SID metrics as gates, or rank iter18 as a control. No formatters, linters, or project-wide tests.
