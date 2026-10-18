# R4_06 (Issue #56) — Unit-normalized cosine distance only for L1 contrastive logits

## 1. Sole mechanism and formula

- **Parent**: canonical Exp1a Resample (`stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE`, `EXPERIMENT_ARM = "resample"`).
- **Mechanism flag**: `R4_06_COSINE_L1`.
- **Formula**: `S'_ij = -(1 - cos(z_i, z_j)) / 0.07 with eps=1e-8`.
- **Changed files vs parent**: `train_rqvae.py`.
- **Trainer SHA256**: `e4f061b1cd2f865d2a55c011410f2000abaf48082295fdc24cdd0d4cc275fa34`.

## 2. TDD RED → GREEN

Full verbatim RED (against the unchanged canonical Resample parent) and GREEN (against this arm's tree) logs, plus the single-delta diff, are recorded in [`tdd_red_green.md`](tdd_red_green.md). Runnable suite: `results/stage2_RQ-VAE/r4_06/test_r4_06.py` (`tests_passed = True`).

## 3. Stage 2 training completion

- **Epochs completed**: `3000` / 3000 (`train_end_3000 = True`).
- **Final checkpoint**: `results/stage2_RQ-VAE/r4_06/rqvae_best.pth` (`sha256 = 3b6233fe9e26f1f94881e75b5fcaa5be7183e2dde18241eb08d7a8419571d036`).
- **Final SIDs**: `results/stage2_RQ-VAE/r4_06/item_sids.json` (`sha256 = e48ee5a9638a744c1c86f673e3f25cfbb28068c935f224e8daa2553e4748ca98`), `sids_for_hgrec.npy`.
- **Canonical reference SID SHA256**: `b08567c7fbaf06e585be37e4c2ceb379f6435e4d36e220c578ae3e6e10265d10`.

## 4. Frozen held-out Hard L1 ER and Stage 3 gate

Evaluated on `results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.npz` (52,855 positive transitions, 10 frozen matched negatives per row, seed 42), column 0 of the exported hard discrete SID only:

| Quantity | Value |
|:---|---:|
| `pos_share` | `0.1059124019` |
| `neg_share` | `0.0179301864` |
| **`ER_L1 = pos_share / neg_share`** | **`5.90693257359924`** |
| Canonical Resample `ER_L1` | `16.38953942232631` |
| `Δ ER_L1` vs canonical | `-10.48260684872707` |
| Strict `ER_L1 > 16.38953942232631` | `False` |
| **`stage3_allowed`** | **`False`** (FAIL — Stage 3 refused) |

## 5. Stage 3 disposition

Gate failed (`ER_L1` did not strictly exceed the canonical Resample baseline `16.38953942232631`). Per the issue contract, Stage 3 was **not** launched, queued, or prepared.
