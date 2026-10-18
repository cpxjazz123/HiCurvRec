# R4_08 (Issue #58) — Encoder-only conflict projection for baseline L1 behavioral gradient

## 1. Sole mechanism and formula

- **Parent**: canonical Exp1a Resample (`stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE`, `EXPERIMENT_ARM = "resample"`).
- **Mechanism flag**: `R4_08_PCGRAD_ENC`.
- **Formula**: `PCGrad projection of g_L1 against g_base on encoder parameters only when <g_L1, g_base> < 0`.
- **Changed files vs parent**: `train_rqvae.py`.
- **Trainer SHA256**: `0999cf94d3376c1a050f8ff5b77605a3643125c92ef32802f7ab285ae6e1a97b`.

## 2. TDD RED → GREEN

Full verbatim RED (against the unchanged canonical Resample parent) and GREEN (against this arm's tree) logs, plus the single-delta diff, are recorded in [`tdd_red_green.md`](tdd_red_green.md). Runnable suite: `results/stage2_RQ-VAE/r4_08/test_r4_08.py` (`tests_passed = True`).

## 3. Stage 2 training completion

- **Epochs completed**: `3000` / 3000 (`train_end_3000 = True`).
- **Final checkpoint**: `results/stage2_RQ-VAE/r4_08/rqvae_best.pth` (`sha256 = e9e8c8a35247db1e06cb52a9a825e869f9338b412e025b306a001f5aa7d0e4b3`).
- **Final SIDs**: `results/stage2_RQ-VAE/r4_08/item_sids.json` (`sha256 = 00829508cfa930ecb66ca9d58158be53e93b687d1218c50ee5cb35679fa9d35d`), `sids_for_hgrec.npy`.
- **Canonical reference SID SHA256**: `b08567c7fbaf06e585be37e4c2ceb379f6435e4d36e220c578ae3e6e10265d10`.

## 4. Frozen held-out Hard L1 ER and Stage 3 gate

Evaluated on `results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.npz` (52,855 positive transitions, 10 frozen matched negatives per row, seed 42), column 0 of the exported hard discrete SID only:

| Quantity | Value |
|:---|---:|
| `pos_share` | `0.0803140668` |
| `neg_share` | `0.0048964147` |
| **`ER_L1 = pos_share / neg_share`** | **`16.40262751159196`** |
| Canonical Resample `ER_L1` | `16.38953942232631` |
| `Δ ER_L1` vs canonical | `+0.01308808926565` |
| Strict `ER_L1 > 16.38953942232631` | `True` |
| **`stage3_allowed`** | **`True`** (PASS — Stage 3 eligible) |

## 5. Stage 3 disposition

Gate passed strictly. Stage 3 was run under the frozen protocol (150 epochs, seed 42, beam 20, `NO_EVAL=True`, `SKIP_TEST=False`, 4-GPU DDP).

| Metric | Value |
|:---|---:|
| `test_recall@5` | `0.0417312279` |
| `test_recall@10` | `0.0629537422` |
| `test_ndcg@5` | `0.0276949892` |
| `test_ndcg@10` | `0.0345476780` |
| `n_eval` | `57439` |

Artifacts: [`test_final.json`](test_final.json), [`stage3_training_metrics.jsonl`](stage3_training_metrics.jsonl).
Exp1a Resample reference (same protocol): `test_recall@10 ≈ 0.0637372`.
