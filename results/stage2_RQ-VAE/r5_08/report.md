# R5_08 (Issue #68) — Late-epoch-only R4-08 conflict projection

## 1. Sole mechanism and formula

- **Parent**: pinned R4-08 (`stage2_RQ-VAE/TIGER_r4_08`, commit `bc96625`).
- **Mechanism flag**: `R5_08_LATE_START_EPOCH`.
- **Formula**: `activate projection only for epoch >= 2250`.
- **Changed files vs parent**: `train_rqvae.py`.
- **Trainer SHA256**: `0c3852728b12ac920c4e523fc803c2d622a15d7e2fb77d18fe1b66f591affd02`.

## 2. TDD RED → GREEN

See [`tdd_red_green.md`](tdd_red_green.md). `tests_passed = True`.

## 3. Stage 2 training completion

- **Epochs completed**: `3000` / 3000.
- **SID SHA256**: `f8770a64605ac1899902f4ba68f10ca39db3be00b67196a0e80ac1a5c4f0cd75`.
- **Canonical Resample SID SHA256**: `b08567c7fbaf06e585be37e4c2ceb379f6435e4d36e220c578ae3e6e10265d10`.

## 4. Frozen held-out Hard L1 ER and Stage 3 gate

| Quantity | Value |
|:---|---:|
| `pos_share` | `0.0803708258` |
| `neg_share` | `0.0051234510` |
| **`ER_L1`** | **`15.68685376661743`** |
| Canonical Resample `ER_L1` | `16.38953942232631` |
| `Δ ER_L1` | `-0.70268565570888` |
| **`stage3_allowed`** | **`False`** (FAIL — Stage 3 refused) |

## 5. Stage 3 disposition

Gate failed. Stage 3 was not launched.
