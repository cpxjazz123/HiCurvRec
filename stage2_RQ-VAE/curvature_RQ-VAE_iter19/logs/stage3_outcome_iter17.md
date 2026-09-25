# Stage3 outcome — iter17

## Observed result

Successful run: `results/stage3_T5Train/curvature_RQ-VAE_iter17/logs/Amazon_2023_Instruments/Sep-26-2026_02-33-02` (started 2026-09-26 02:33:02).

The run completed all **150 epochs**. At epoch 150, the logged train loss was **1.8663109588623046**; the lowest logged train loss was **1.865799880027771** at epoch 147. The final evaluation reports `n_eval=57439` and these exact metrics:

| Metric | Value |
|---|---:|
| `test_recall@5` | 0.038649697940423756 |
| `test_recall@10` | 0.059071362662999005 |
| `test_ndcg@5` | 0.025837394816762203 |
| `test_ndcg@10` | 0.032386492092076745 |

`test_final.json` names the best checkpoint as `results/stage3_T5Train/curvature_RQ-VAE_iter17/ckpt/Amazon_2023_Instruments/Sep-26-2026_02-33-02/HG_Rec_best.pth`.

The strict target is `test_recall@10 > 0.065`. **Outcome: FAIL.** The shortfall from 0.065 is **0.005928637337000995** (0.065 − 0.059071362662999005); equality would not pass the strict target.

## Comparison (observed)

Exact `test_recall@10` values from the respective final evaluation JSON files:

| Iteration | `test_recall@10` | Iter17 difference |
|---|---:|---:|
| iter11 | 0.05976775361688052 | -0.000696390953881515 |
| iter16 | 0.05687773115827226 | +0.002193631504726745 |
| iter17 | 0.059071362662999005 | — |

Thus iter17 is below iter11 and above iter16 on this metric; it misses the hard target in either comparison. These are observed run-to-run score differences, not evidence by themselves that the Stage2 mechanism caused them.

## Interpretation (limited)

Iter17's Stage2 hypothesis describes an inverse-curvature layerwise AdamW learning-rate multiplier. The recorded Stage2 report says the registered direct-effect checks passed and classifies the SID geometry as aligned, while explicitly treating those checks and geometry as descriptive rather than a gate. Stage3 is the downstream decision: the completed iter17 run does not meet `test_recall@10 > 0.065`. The scores establish the observed comparison above, but do not isolate a causal effect of the optimizer change.

The earlier aborted 02:29 Stage3 attempt is excluded; it is not an evaluation result and is not analyzed here.

## Memory

- Stage2 direct effects and SID geometry are descriptive evidence only; neither is a substitute for the completed Stage3 outcome.
- Do not use `hitrate@50` or SID geometry as a gate. Stage2 geometry must not gate Stage3; the downstream hard target decides.
- In this iteration, the inverse-curvature AdamW variant missed the target; its observed `test_recall@10` was below iter11 and above iter16. This records outcomes, not a causal attribution.

## Source artifacts

- `results/stage3_T5Train/curvature_RQ-VAE_iter17/logs/Amazon_2023_Instruments/Sep-26-2026_02-33-02/test_final.json`
- `results/stage3_T5Train/curvature_RQ-VAE_iter17/logs/Amazon_2023_Instruments/Sep-26-2026_02-33-02/HG_Rec.log`
- `results/stage3_T5Train/curvature_RQ-VAE_iter17/logs/Amazon_2023_Instruments/Sep-26-2026_02-33-02/training_metrics.jsonl`
- `stage2_RQ-VAE/curvature_RQ-VAE_iter17/logs/sid_geometry_iter17.md`
- `stage2_RQ-VAE/curvature_RQ-VAE_iter17/logs/hypothesis_iter17.md`
- `results/stage3_T5Train/curvature_RQ-VAE_iter11/logs/Amazon_2023_Instruments/Sep-25-2026_05-25-18/test_final.json`
- `results/stage3_T5Train/curvature_RQ-VAE_iter16/logs/Amazon_2023_Instruments/Sep-26-2026_00-03-22/test_final.json`
