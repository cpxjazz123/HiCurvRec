# Stage3 outcome — iter18

## Observed result

Successful run: `results/stage3_T5Train/curvature_RQ-VAE_iter18/logs/Amazon_2023_Instruments/Sep-26-2026_04-45-57` (started 2026-09-26 04:45:57). The run completed all **150 epochs**. Epoch 150 train loss was **1.8682917833328248**; the lowest logged train loss was **1.8669320297241212** at epoch 149. The final evaluation reports `n_eval=57439` and these exact metrics:

| Metric | Value |
|---|---:|
| `test_recall@5` | 0.040460314420515675 |
| `test_recall@10` | 0.05988962203380978 |
| `test_ndcg@5` | 0.02697174940916111 |
| `test_ndcg@10` | 0.03323035190128754 |

`test_final.json` names `results/stage3_T5Train/curvature_RQ-VAE_iter18/ckpt/Amazon_2023_Instruments/Sep-26-2026_04-45-57/HG_Rec_best.pth` as the evaluated checkpoint.

The strict target is `test_recall@10 > 0.065`. **Outcome: FAIL.** The shortfall from 0.065 is **0.005110377966190224**; equality would not pass the strict target.

## Comparison (observed)

Exact `test_recall@10` values from the final evaluation artifacts:

| Iteration | `test_recall@10` | Iter18 difference |
|---|---:|---:|
| iter11 | 0.05976775361688052 | +0.00012186841692925915 |
| iter16 | 0.05687773115827226 | +0.0030118908755375207 |
| iter17 | 0.059071362662999005 | +0.0008182593708107727 |
| iter18 | 0.05988962203380978 | — |

These are observed run-to-run score differences, not causal estimates.

## Interpretation (limited)

Iter18 tested P18B, fixed per-layer AdamW `β₂` derived from initial curvature while retaining cyclic curvature. Its registered MVG checks passed, Stage2 completed 100,000 steps, and the four-token SID export passed `SID_WIRING_PASS`. The completed Stage3 run missed the fixed downstream target. The score alone does not establish that P18B caused this result, that the mechanism has no effect, or that the target is unreachable.

Pipeline identity caveat: the launch audit configures the iter18 input and `iter18_initial_curvature_beta2`; the first `training_metrics.jsonl` event confirms `world_size=4` and the iter18 `item_sids.json` `code_path`, but records `variant="unknown_variant"`. The input path and completed run are evidenced; the variant label remains unverified metadata. The launcher also records nonfatal NCCL wrapper warnings, and process exit records that `destroy_process_group()` was not called. The final test event exists, and targeted searches found no `Traceback`, `RuntimeError`, `ERROR`, or `FAIL` markers in the Stage3 logs.

## Source artifacts

- `results/stage3_T5Train/curvature_RQ-VAE_iter18/logs/Amazon_2023_Instruments/Sep-26-2026_04-45-57/test_final.json`
- `results/stage3_T5Train/curvature_RQ-VAE_iter18/logs/Amazon_2023_Instruments/Sep-26-2026_04-45-57/HG_Rec.log`
- `results/stage3_T5Train/curvature_RQ-VAE_iter18/logs/Amazon_2023_Instruments/Sep-26-2026_04-45-57/training_metrics.jsonl`
- `stage2_RQ-VAE/curvature_RQ-VAE_iter18/logs/stage3_launch_iter18.log`
- `stage2_RQ-VAE/curvature_RQ-VAE_iter18/logs/sid_geometry_iter18.md`
- `stage2_RQ-VAE/curvature_RQ-VAE_iter18/logs/hypothesis_iter18.md`
- `results/stage3_T5Train/curvature_RQ-VAE_iter11/logs/Amazon_2023_Instruments/Sep-25-2026_05-25-18/test_final.json`
- `results/stage3_T5Train/curvature_RQ-VAE_iter16/logs/Amazon_2023_Instruments/Sep-26-2026_00-03-22/test_final.json`
- `results/stage3_T5Train/curvature_RQ-VAE_iter17/logs/Amazon_2023_Instruments/Sep-26-2026_02-33-02/test_final.json`
