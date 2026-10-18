# R5_01 (Issue #61) — Reverse PCGrad: protect L1 behavior against base gradient

## 1. Sole mechanism and formula

- **Parent**: pinned R4-08 (`stage2_RQ-VAE/TIGER_r4_08`, commit `bc96625`).
- **Mechanism flag**: `R5_01_REVERSE_PCGRAD`.
- **Formula**: `project g_base against g_L1 when conflict`.
- **Changed files vs parent**: `train_rqvae.py`.
- **Trainer SHA256**: `2a94810c212cf9f556786531e286d2b1ac06621c4716b95be47a10833945069e`.

## 2. TDD RED → GREEN

See [`tdd_red_green.md`](tdd_red_green.md). `tests_passed = True`.

## 3. Stage 2 training completion

- **Epochs completed**: `3000` / 3000.
- **SID SHA256**: `c6890564f1a72046b881f419cd10d4b36e0e21142e506656a670b147b1fbed00`.
- **Canonical Resample SID SHA256**: `b08567c7fbaf06e585be37e4c2ceb379f6435e4d36e220c578ae3e6e10265d10`.

## 4. Frozen held-out Hard L1 ER and Stage 3 gate

| Quantity | Value |
|:---|---:|
| `pos_share` | `0.0796329581` |
| `neg_share` | `0.0049153344` |
| **`ER_L1`** | **`16.20092378752887`** |
| Canonical Resample `ER_L1` | `16.38953942232631` |
| `Δ ER_L1` | `-0.18861563479744` |
| **`stage3_allowed`** | **`False`** (FAIL — Stage 3 refused) |

## 5. Stage 3 disposition

Gate failed. Stage 3 was not launched.
