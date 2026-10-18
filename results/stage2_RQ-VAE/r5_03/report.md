# R5_03 (Issue #63) — Quantization-only reference gradient for original L1 PCGrad

## 1. Sole mechanism and formula

- **Parent**: pinned R4-08 (`stage2_RQ-VAE/TIGER_r4_08`, commit `bc96625`).
- **Mechanism flag**: `R5_03_QUANT_REF`.
- **Formula**: `project g_L1 against g_quant`.
- **Changed files vs parent**: `train_rqvae.py`.
- **Trainer SHA256**: `b51d190da69149b10e79aa8ff12d342240d9b66b83f599b59d607001f15c3057`.

## 2. TDD RED → GREEN

See [`tdd_red_green.md`](tdd_red_green.md). `tests_passed = True`.

## 3. Stage 2 training completion

- **Epochs completed**: `3000` / 3000.
- **SID SHA256**: `656ea9ceca603b17b486b112e7eaebe4abf8e1aa1d9212679cb4b0f29fbd774d`.
- **Canonical Resample SID SHA256**: `b08567c7fbaf06e585be37e4c2ceb379f6435e4d36e220c578ae3e6e10265d10`.

## 4. Frozen held-out Hard L1 ER and Stage 3 gate

| Quantity | Value |
|:---|---:|
| `pos_share` | `0.0808059786` |
| `neg_share` | `0.0051215590` |
| **`ER_L1`** | **`15.77761359438493`** |
| Canonical Resample `ER_L1` | `16.38953942232631` |
| `Δ ER_L1` | `-0.61192582794138` |
| **`stage3_allowed`** | **`False`** (FAIL — Stage 3 refused) |

## 5. Stage 3 disposition

Gate failed. Stage 3 was not launched.
