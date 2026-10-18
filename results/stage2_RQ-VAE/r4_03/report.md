# R4_03 (Issue #53) — Inverse-anchor-frequency weighting of baseline L1 contrastive rows

## 1. Sole mechanism and formula

- **Parent**: canonical Exp1a Resample (`stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE`, `EXPERIMENT_ARM = "resample"`).
- **Mechanism flag**: `R4_03_ANCHOR_WEIGHTING`.
- **Formula**: `w(a) = (1 + f(a))**-0.5; L1_new = sum_i w(a_i) CE_i / sum_i w(a_i)`.
- **Changed files vs parent**: `train_rqvae.py`.
- **Trainer SHA256**: `8abed4b75a89a6540934d38b76c9946bb61e484a26a227d2d0c483e6637c5d55`.

## 2. TDD RED → GREEN

Full verbatim RED (against the unchanged canonical Resample parent) and GREEN (against this arm's tree) logs, plus the single-delta diff, are recorded in [`tdd_red_green.md`](tdd_red_green.md). Runnable suite: `results/stage2_RQ-VAE/r4_03/test_r4_03.py` (`tests_passed = True`).

## 3. Stage 2 training completion

- **Epochs completed**: `3000` / 3000 (`train_end_3000 = True`).
- **Final checkpoint**: `results/stage2_RQ-VAE/r4_03/rqvae_best.pth` (`sha256 = db44c3d30f4ab82fe2e8be99a373bf037ab3cf58d6077f2a8bc76936f3f11bc3`).
- **Final SIDs**: `results/stage2_RQ-VAE/r4_03/item_sids.json` (`sha256 = cc97adbd7815135a38271c481f2352a6f9099c2609630d5007222ef4193f0902`), `sids_for_hgrec.npy`.
- **Canonical reference SID SHA256**: `b08567c7fbaf06e585be37e4c2ceb379f6435e4d36e220c578ae3e6e10265d10`.

## 4. Frozen held-out Hard L1 ER and Stage 3 gate

Evaluated on `results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.npz` (52,855 positive transitions, 10 frozen matched negatives per row, seed 42), column 0 of the exported hard discrete SID only:

| Quantity | Value |
|:---|---:|
| `pos_share` | `0.0748462776` |
| `neg_share` | `0.0048869549` |
| **`ER_L1 = pos_share / neg_share`** | **`15.31552458381727`** |
| Canonical Resample `ER_L1` | `16.38953942232631` |
| `Δ ER_L1` vs canonical | `-1.07401483850904` |
| Strict `ER_L1 > 16.38953942232631` | `False` |
| **`stage3_allowed`** | **`False`** (FAIL — Stage 3 refused) |

## 5. Stage 3 disposition

Gate failed (`ER_L1` did not strictly exceed the canonical Resample baseline `16.38953942232631`). Per the issue contract, Stage 3 was **not** launched, queued, or prepared.
