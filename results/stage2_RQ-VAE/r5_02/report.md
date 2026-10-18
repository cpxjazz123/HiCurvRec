# R5_02 (Issue #62) — Reconstruction-only reference gradient for original L1 PCGrad

## 1. Sole mechanism and formula

- **Parent**: pinned R4-08 (`stage2_RQ-VAE/TIGER_r4_08`, commit `bc96625`).
- **Mechanism flag**: `R5_02_RECON_REF`.
- **Formula**: `project g_L1 against g_recon`.
- **Changed files vs parent**: `train_rqvae.py`.
- **Trainer SHA256**: `3e2b99efdafc8f2780a177a81c7f04286d0949c39c5b2275d19210fe6a476692`.

## 2. TDD RED → GREEN

See [`tdd_red_green.md`](tdd_red_green.md). `tests_passed = True`.

## 3. Stage 2 training completion

- **Epochs completed**: `3000` / 3000.
- **SID SHA256**: `607e932b6ad3bcdaf36e61a0aee0c0a2d2c118692bab2440f4f2303c572e288f`.
- **Canonical Resample SID SHA256**: `b08567c7fbaf06e585be37e4c2ceb379f6435e4d36e220c578ae3e6e10265d10`.

## 4. Frozen held-out Hard L1 ER and Stage 3 gate

| Quantity | Value |
|:---|---:|
| `pos_share` | `0.0790653675` |
| `neg_share` | `0.0048945228` |
| **`ER_L1`** | **`16.15384615384615`** |
| Canonical Resample `ER_L1` | `16.38953942232631` |
| `Δ ER_L1` | `-0.23569326848015` |
| **`stage3_allowed`** | **`False`** (FAIL — Stage 3 refused) |

## 5. Stage 3 disposition

Gate failed. Stage 3 was not launched.
