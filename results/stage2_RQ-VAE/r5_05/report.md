# R5_05 (Issue #65) — Half-strength conflict projection alpha=0.5

## 1. Sole mechanism and formula

- **Parent**: pinned R4-08 (`stage2_RQ-VAE/TIGER_r4_08`, commit `bc96625`).
- **Mechanism flag**: `R5_05_ALPHA`.
- **Formula**: `g'_L1 = g_L1 - 0.5 * proj_component`.
- **Changed files vs parent**: `train_rqvae.py`.
- **Trainer SHA256**: `21f209d476202bb9258342a0743fd5bae21280363c919c2d7dec185b595bc15c`.

## 2. TDD RED → GREEN

See [`tdd_red_green.md`](tdd_red_green.md). `tests_passed = True`.

## 3. Stage 2 training completion

- **Epochs completed**: `3000` / 3000.
- **SID SHA256**: `41812f406047bac4df215848ce3db04a538c084614efa19867ca09fb9e81f7dd`.
- **Canonical Resample SID SHA256**: `b08567c7fbaf06e585be37e4c2ceb379f6435e4d36e220c578ae3e6e10265d10`.

## 4. Frozen held-out Hard L1 ER and Stage 3 gate

| Quantity | Value |
|:---|---:|
| `pos_share` | `0.0810897739` |
| `neg_share` | `0.0050572320` |
| **`ER_L1`** | **`16.03441825664048`** |
| Canonical Resample `ER_L1` | `16.38953942232631` |
| `Δ ER_L1` | `-0.35512116568583` |
| **`stage3_allowed`** | **`False`** (FAIL — Stage 3 refused) |

## 5. Stage 3 disposition

Gate failed. Stage 3 was not launched.
