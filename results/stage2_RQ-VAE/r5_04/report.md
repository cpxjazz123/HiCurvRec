# R5_04 (Issue #64) — Per-encoder-layer blockwise PCGrad

## 1. Sole mechanism and formula

- **Parent**: pinned R4-08 (`stage2_RQ-VAE/TIGER_r4_08`, commit `bc96625`).
- **Mechanism flag**: `R5_04_BLOCKWISE`.
- **Formula**: `PCGrad per Linear weight+bias block`.
- **Changed files vs parent**: `train_rqvae.py`.
- **Trainer SHA256**: `30d0530b24fa288b27f5072cab3367b3d801e06171f720f73c45714cbc5bc8a9`.

## 2. TDD RED → GREEN

See [`tdd_red_green.md`](tdd_red_green.md). `tests_passed = True`.

## 3. Stage 2 training completion

- **Epochs completed**: `3000` / 3000.
- **SID SHA256**: `864de2b5846e668ecb0f04212ced33a9625ebde929e7d439da39a398f4ab7e0b`.
- **Canonical Resample SID SHA256**: `b08567c7fbaf06e585be37e4c2ceb379f6435e4d36e220c578ae3e6e10265d10`.

## 4. Frozen held-out Hard L1 ER and Stage 3 gate

| Quantity | Value |
|:---|---:|
| `pos_share` | `0.0761517359` |
| `neg_share` | `0.0047866805` |
| **`ER_L1`** | **`15.90909090909091`** |
| Canonical Resample `ER_L1` | `16.38953942232631` |
| `Δ ER_L1` | `-0.48044851323540` |
| **`stage3_allowed`** | **`False`** (FAIL — Stage 3 refused) |

## 5. Stage 3 disposition

Gate failed. Stage 3 was not launched.
