# R4 Round (Issues #51–#60) — Stage 2 gate summary

Parent: Exp1a Resample (`TIGER_BEHAVIOR_SAMPLE_RQ-VAE` / `resample`).
Metric: frozen Hard L1 ER on `behavior_pairs_valid_s42.npz`, SID column 0.
Baseline ER_L1 = `16.38953942232631`.
Gate: `stage3_allowed = tests_passed && train_end_3000 && hashes_valid && pos_neg_support_valid && isfinite(ER_L1) && neg_share>0 && (ER_L1 > baseline)`.

| Arm | Issue | ER_L1 | Δ vs baseline | stage3_allowed |
|:---|---:|---:|---:|:---:|
| `r4_01` | #51 | 14.559859 | -1.829680 | `False` |
| `r4_02` | #52 | 16.088133 | -0.301406 | `False` |
| `r4_03` | #53 | 15.315525 | -1.074015 | `False` |
| `r4_04` | #54 | 15.607264 | -0.782275 | `False` |
| `r4_05` | #55 | 15.146134 | -1.243405 | `False` |
| `r4_06` | #56 | 5.906933 | -10.482607 | `False` |
| `r4_07` | #57 | 15.975565 | -0.413975 | `False` |
| `r4_08` | #58 | 16.402628 | +0.013088 | `True` |
| `r4_09` | #59 | 1.000000 | -15.389539 | `False` |
| `r4_10` | #60 | 15.939658 | -0.449881 | `False` |

## Stage 3

Only `r4_08` (#58) passed the gate and was launched (4-GPU DDP, 150 epochs, seed 42, beam 20).

| Metric | `r4_08` |
|:---|---:|
| `test_recall@5` | `0.04173122791134943` |
| `test_recall@10` | `0.06295374223088843` |
| `test_ndcg@5` | `0.02769498921319816` |
| `test_ndcg@10` | `0.03454767796170093` |
| `n_eval` | `57439` |

Per-arm deliverables live under `results/stage2_RQ-VAE/r4_NN/` (`gate.json`, `stage2_metrics.json`, `report.md`, `tdd_red_green.md`, `item_sids.json`).
