# R5_09 (Issue #69) — EMA-smoothed base gradient as PCGrad reference

## 1. Sole mechanism and formula

- **Parent**: pinned R4-08 (`stage2_RQ-VAE/TIGER_r4_08`, commit `bc96625`).
- **Mechanism flag**: `R5_09_EMA_DECAY`.
- **Formula**: `m_t=0.9 m + 0.1 g_base as reference`.
- **Changed files vs parent**: `train_rqvae.py`.
- **Trainer SHA256**: `9d5691752a9119f030fba9d16c9ed92b5eaecbad036db59bcf04d8e2a70cba9b`.

## 2. TDD RED → GREEN

See [`tdd_red_green.md`](tdd_red_green.md). `tests_passed = True`.

## 3. Stage 2 training completion

- **Epochs completed**: `3000` / 3000.
- **SID SHA256**: `10b2f7cdb4910d1c78656f5192327129ecf1e03a77ac6da459dd206e616f3c66`.
- **Canonical Resample SID SHA256**: `b08567c7fbaf06e585be37e4c2ceb379f6435e4d36e220c578ae3e6e10265d10`.

## 4. Frozen held-out Hard L1 ER and Stage 3 gate

| Quantity | Value |
|:---|---:|
| `pos_share` | `0.0783653391` |
| `neg_share` | `0.0050723678` |
| **`ER_L1`** | **`15.44945915703096`** |
| Canonical Resample `ER_L1` | `16.38953942232631` |
| `Δ ER_L1` | `-0.94008026529535` |
| **`stage3_allowed`** | **`False`** (FAIL — Stage 3 refused) |

## 5. Stage 3 disposition

Gate failed. Stage 3 was not launched.
