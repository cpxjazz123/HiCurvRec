# R4_09 (Issue #59) — Target-branch stop-gradient in original L1 InfoNCE

## 1. Sole mechanism and formula

- **Parent**: canonical Exp1a Resample (`stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE`, `EXPERIMENT_ARM = "resample"`).
- **Mechanism flag**: `R4_09_TARGET_DETACH`.
- **Formula**: `candidates = residuals[0, batch_size:].detach() at L1 only`.
- **Changed files vs parent**: `train_rqvae.py`.
- **Trainer SHA256**: `aa092d4dc06fd3995c4019e9721be47e65e45c1cab82d8190e970733c279fca6`.

## 2. TDD RED → GREEN

Full verbatim RED (against the unchanged canonical Resample parent) and GREEN (against this arm's tree) logs, plus the single-delta diff, are recorded in [`tdd_red_green.md`](tdd_red_green.md). Runnable suite: `results/stage2_RQ-VAE/r4_09/test_r4_09.py` (`tests_passed = True`).

## 3. Stage 2 training completion

- **Epochs completed**: `3000` / 3000 (`train_end_3000 = True`).
- **Final checkpoint**: `results/stage2_RQ-VAE/r4_09/rqvae_best.pth` (`sha256 = 04f04f16e0b8e52e50933081db2ef281d922beae0e8eadde030702fb9c9cac26`).
- **Final SIDs**: `results/stage2_RQ-VAE/r4_09/item_sids.json` (`sha256 = f0d852fcfb911e8ad673aa98eec803b9b8360f39c1a939c28e19c409b16de642`), `sids_for_hgrec.npy`.
- **Canonical reference SID SHA256**: `b08567c7fbaf06e585be37e4c2ceb379f6435e4d36e220c578ae3e6e10265d10`.

## 4. Frozen held-out Hard L1 ER and Stage 3 gate

Evaluated on `results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.npz` (52,855 positive transitions, 10 frozen matched negatives per row, seed 42), column 0 of the exported hard discrete SID only:

| Quantity | Value |
|:---|---:|
| `pos_share` | `1.0000000000` |
| `neg_share` | `1.0000000000` |
| **`ER_L1 = pos_share / neg_share`** | **`1.00000000000000`** |
| Canonical Resample `ER_L1` | `16.38953942232631` |
| `Δ ER_L1` vs canonical | `-15.38953942232631` |
| Strict `ER_L1 > 16.38953942232631` | `False` |
| **`stage3_allowed`** | **`False`** (FAIL — Stage 3 refused) |

## 5. Stage 3 disposition

Gate failed (`ER_L1` did not strictly exceed the canonical Resample baseline `16.38953942232631`). Per the issue contract, Stage 3 was **not** launched, queued, or prepared.
