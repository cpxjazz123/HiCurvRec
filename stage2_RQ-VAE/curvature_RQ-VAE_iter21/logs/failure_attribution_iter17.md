# Failure attribution — iter17

## Decision

**Selected outcome: `TRUE_MECHANISM_FAIL` — qualified as a downstream target miss, not proof of a causal mechanism failure.** The successful Stage3 run completed on the registered iter17 input and missed the strict `test_recall@10 > 0.065` target. The implementation, registered activation checks, descriptive geometry assessment, and successful-run pipeline checks below show no evidenced implementation, activation, geometry, or pipeline failure. One downstream score cannot isolate the optimizer change's causal contribution.

## Evidence and attribution checks

### Implementation — pass; `IMPLEMENTATION_FAIL` not supported

The iter17 four-gate MVG log reports `MVG PASS` on a `(640, 768)` batch with 640 active pairs. Its final audit records finite, positive `c_l` and `m_l` values at steps 0, 50,000, and 100,000; `mean(m_l(0))=1`; and midpoint multipliers `[0.399268, 0.082085, 0.073250]` versus step-0 `[2.151887, 0.448100, 0.400013]`. The Stage2 training log reaches global step 100,000, saves `rqvae_best.pth` at that step, and reports `train done`. These support the optimizer-specific implementation and completion checks, rather than an evidenced `IMPLEMENTATION_FAIL`. **Provenance caveat:** `mvg_check_iter17.log` warns that iter17 `layer_norms.json` is missing and records use of iter1-derived residual scales `[0.001, 0.932889, 1.0]`. These are residual-layer normalization inputs, separate from the optimizer's `m_l`/per-layer learning-rate calculation (`curvature_RQ-VAE.py` Step5); this caveat does not alter the recorded optimizer-multiplier calculations or direct-effect values. It qualifies the provenance of the residual-scale inputs, but the available evidence does not connect that fallback to the Stage3 result.

### Activation — all registered direct effects pass; `ACTIVATION_FAIL` not supported

The pre-registered effects in `hypothesis_iter17.md` are all observed in `mvg_check_iter17.log` and mapped by `sid_geometry_iter17.md`:

- **DE-1, cyclic curvature / multiplier schedule:** finite positive values at all three audited steps, initial multiplier mean 1, and the specified midpoint-versus-initial change (above `1e-3`).
- **DE-2, per-layer updates after five ON steps:** relative updates are layer0 `0.04763167998820698`, layer1 `0.10315319816669052`, layer2 `0.1008355832236253`, each above `1e-7`.
- **DE-3, same-seed ON/OFF difference after 200 steps:** `L_on=2.54342532`, `L_off=2.46642065`, `abs(delta)=0.07700`, above `1e-6`.

The MVG calls these the five-step `DE-3` and 200-step `DE-4` because it has an extra internal gradient gate; the values correspond to hypothesis DE-2 and DE-3. The evidence supports activation of the registered optimizer effects, not that they must improve downstream recall.

### Geometry — descriptive report is `ALIGNED`; `GEOMETRY_FAIL` not supported

The registered three-token geometry report classifies the proxy as `ALIGNED`: weighted `H(L1|L0)=5.5939` bits meets its descriptive threshold of 5.45; three-token collision rate is 6.08% (1,495 collision rows); and the report notes an iter11-like coarse/fine branching ratio. The training summary reports `full_gini=0.0586`, layer Ginis `[0.1904, 0.2873, 0.3977]`, and 23,092 unique three-token SIDs. This is descriptive alignment only, not a performance gate or proof of causal geometric improvement. The report also states that an L0 oracle is unavailable. `hitrate@50` is neither used as an oracle nor treated as a gate. Under the current Stage2 policy, no SID metric gates Stage3.

### Pipeline — successful-run checks pass; `PIPELINE_FAIL` not supported

The successful attempt is the run directory `Sep-26-2026_02-33-02`; the earlier aborted 02:29 attempt is excluded. The wrapper `stage2_RQ-VAE/curvature_RQ-VAE_iter17/scripts/run_stage3_iter17.py` points Stage3 at iter17 `item_sids.json` and hard-codes the required NCCL settings (`NCCL_IB_DISABLE=1`, `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=1`, `NCCL_TIMEOUT=3600`, `TORCH_NCCL_BLOCKING_WAIT=1`). The launch audit states that only the second, corrected 4-rank/port-50201 attempt supplies the result. Its `training_metrics.jsonl` records `world_size=4` and `code_path=/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter17/item_sids.json`; Stage2's export log reports four-token SID NPY shape `(24587, 4)`, matching JSON with 24,587 items, and `SID_WIRING_PASS`. The final Stage3 evaluation names `HG_Rec_best.pth` under this successful run's checkpoint directory and completed all 150 epochs. **Metadata caveat:** the run's `train_start` event records `variant="unknown_variant"` even though the wrapper sets `RQVAE_VARIANT="iter17_riemannian_adam_1_over_c"`. This is metadata only: the code path, successful four-rank run, SID export/counts, and final checkpoint/test evidence identify the actual input and evaluated run; the available evidence does not show that the variant metadata altered runtime behavior. Therefore this caveat does not establish `PIPELINE_FAIL`.

## Observed Stage3 outcome (successful run only)

`test_final.json` reports `n_eval=57439` and the following exact values:

| Metric | Value |
|---|---:|
| `test_recall@5` | 0.038649697940423756 |
| `test_recall@10` | 0.059071362662999005 |
| `test_ndcg@5` | 0.025837394816762203 |
| `test_ndcg@10` | 0.032386492092076745 |

The hard target is strict `test_recall@10 > 0.065`; observed recall@10 misses by `0.005928637337000995`. Exact comparison values are iter11 `0.05976775361688052` (iter17 difference `-0.000696390953881515`) and iter16 `0.05687773115827226` (iter17 difference `+0.002193631504726745`). Iter17 falls below iter11 and above iter16; these are observed comparisons, not causal estimates.

## Alternative attribution classes

- **`IMPLEMENTATION_FAIL` — not selected:** the MVG passed and Stage2 reached the 100,000-step completion/checkpoint; no implementation defect accounting for the outcome is evidenced.
- **`ACTIVATION_FAIL` — not selected:** all three pre-registered direct effects passed with the reported thresholds and values.
- **`GEOMETRY_FAIL` — not selected:** the registered descriptive report says `ALIGNED`. This label is not elevated into a gate; neither these proxies nor missing L0-oracle evidence establish downstream causality.
- **`PIPELINE_FAIL` — not selected:** the successful corrected launch used the iter17 four-token SID input, completed 150 epochs, and generated the cited final checkpoint and evaluation; the aborted earlier attempt is excluded.
- **`TRUE_MECHANISM_FAIL` — selected with qualification:** factual finding: this completed, correctly wired run missed the fixed downstream target. Inference only: the optimizer mechanism did not produce enough downstream gain to meet that target in this run. The evidence does **not** prove that the optimizer change caused the score, that the mechanism has no effect, or that the target is unreachable.

## Evidence sources

- `logs/mvg_check_iter17.log`; `logs/train_migrated.log`; `logs/sid_geometry_iter17.md`; `logs/hypothesis_iter17.md`
- `scripts/run_stage3_iter17.py`; `logs/stage3_launch_iter17.log`
- `results/stage3_T5Train/curvature_RQ-VAE_iter17/logs/Amazon_2023_Instruments/Sep-26-2026_02-33-02/{training_metrics.jsonl,HG_Rec.log,test_final.json}`
- `stage2_RQ-VAE/curvature_RQ-VAE_iter17/logs/stage3_outcome_iter17.md` (outcome context and exact historical comparison values)
