# R5_06 (Issue #66) — Norm-preserving post-PCGrad L1 behavior gradient

## 1. Sole mechanism and formula

- **Parent**: pinned R4-08 (`stage2_RQ-VAE/TIGER_r4_08`, commit `bc96625`).
- **Mechanism flag**: `R5_06_NORM_PRESERVE`.
- **Formula**: `rescale projected g_L1 to original norm`.
- **Changed files vs parent**: `train_rqvae.py`.
- **Trainer SHA256**: `4520a2330bc418c04b6d253db3743195db3e7aca1821a0ab3803a4284e1a6307`.

## 2. TDD RED → GREEN

See [`tdd_red_green.md`](tdd_red_green.md). `tests_passed = True`.

## 3. Stage 2 training completion

- **Epochs completed**: `3000` / 3000.
- **SID SHA256**: `41490b1555748c3d960f8eaffd10b7f05fe8f5e5fb8613e4c3192072aa5a434b`.
- **Canonical Resample SID SHA256**: `b08567c7fbaf06e585be37e4c2ceb379f6435e4d36e220c578ae3e6e10265d10`.

## 4. Frozen held-out Hard L1 ER and Stage 3 gate

| Quantity | Value |
|:---|---:|
| `pos_share` | `0.0771166399` |
| `neg_share` | `0.0050212846` |
| **`ER_L1`** | **`15.35795026375282`** |
| Canonical Resample `ER_L1` | `16.38953942232631` |
| `Δ ER_L1` | `-1.03158915857348` |
| **`stage3_allowed`** | **`False`** (FAIL — Stage 3 refused) |

## 5. Stage 3 disposition

Gate failed. Stage 3 was not launched.
