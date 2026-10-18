# R5_07 (Issue #67) — Strong-conflict-only cosine trigger

## 1. Sole mechanism and formula

- **Parent**: pinned R4-08 (`stage2_RQ-VAE/TIGER_r4_08`, commit `bc96625`).
- **Mechanism flag**: `R5_07_COS_THRESHOLD`.
- **Formula**: `project only if cos < -0.2`.
- **Changed files vs parent**: `train_rqvae.py`.
- **Trainer SHA256**: `00fcb21f3511d59275a4d2f635279ee118f262ed8302a16a4196772ebe64c33e`.

## 2. TDD RED → GREEN

See [`tdd_red_green.md`](tdd_red_green.md). `tests_passed = True`.

## 3. Stage 2 training completion

- **Epochs completed**: `3000` / 3000.
- **SID SHA256**: `ec04f48f7086681e375f8250d4063b106d0f990558b363508793f34f3d5737e3`.
- **Canonical Resample SID SHA256**: `b08567c7fbaf06e585be37e4c2ceb379f6435e4d36e220c578ae3e6e10265d10`.

## 4. Frozen held-out Hard L1 ER and Stage 3 gate

| Quantity | Value |
|:---|---:|
| `pos_share` | `0.0787437329` |
| `neg_share` | `0.0048926308` |
| **`ER_L1`** | **`16.09435421500386`** |
| Canonical Resample `ER_L1` | `16.38953942232631` |
| `Δ ER_L1` | `-0.29518520732244` |
| **`stage3_allowed`** | **`False`** (FAIL — Stage 3 refused) |

## 5. Stage 3 disposition

Gate failed. Stage 3 was not launched.
