# R5_10 (Issue #70) — Inter-level PCGrad against L2+L3

## 1. Sole mechanism and formula

- **Parent**: pinned R4-08 (`stage2_RQ-VAE/TIGER_r4_08`, commit `bc96625`).
- **Mechanism flag**: `R5_10_INTERLEVEL`.
- **Formula**: `project g_L1 against g_other`.
- **Changed files vs parent**: `train_rqvae.py`.
- **Trainer SHA256**: `2291f7fb7a4053fa3acf2d553956ceb925fd014141991cf6e07bef8f686079bc`.

## 2. TDD RED → GREEN

See [`tdd_red_green.md`](tdd_red_green.md). `tests_passed = True`.

## 3. Stage 2 training completion

- **Epochs completed**: `3000` / 3000.
- **SID SHA256**: `b2157e4cdfb1b618a977118b8b2128f8d11c17739cba9ba684c7c276e1e6d809`.
- **Canonical Resample SID SHA256**: `b08567c7fbaf06e585be37e4c2ceb379f6435e4d36e220c578ae3e6e10265d10`.

## 4. Frozen held-out Hard L1 ER and Stage 3 gate

| Quantity | Value |
|:---|---:|
| `pos_share` | `0.0790653675` |
| `neg_share` | `0.0050080409` |
| **`ER_L1`** | **`15.78768417075935`** |
| Canonical Resample `ER_L1` | `16.38953942232631` |
| `Δ ER_L1` | `-0.60185525156696` |
| **`stage3_allowed`** | **`False`** (FAIL — Stage 3 refused) |

## 5. Stage 3 disposition

Gate failed. Stage 3 was not launched.
