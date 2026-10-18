# R4_10 (Issue #60) — Train-target-frequency-weighted L1 KMeans initialization only

## 1. Sole mechanism and formula

- **Parent**: canonical Exp1a Resample (`stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE`, `EXPERIMENT_ARM = "resample"`).
- **Mechanism flag**: `R4_10_WEIGHTED_KMEANS`.
- **Formula**: `L1 KMeans fit with sample_weight w_i = sqrt(1 + count_train_target(i)), mean-normalised`.
- **Changed files vs parent**: `model/layers.py, model/model.py, train_rqvae.py`.
- **Trainer SHA256**: `d022fb8cc412c8e6257f7cb872b179feba4d85aa3704ea680ed07717d5bdcd4a`.

## 2. TDD RED → GREEN

Full verbatim RED (against the unchanged canonical Resample parent) and GREEN (against this arm's tree) logs, plus the single-delta diff, are recorded in [`tdd_red_green.md`](tdd_red_green.md). Runnable suite: `results/stage2_RQ-VAE/r4_10/test_r4_10.py` (`tests_passed = True`).

## 3. Stage 2 training completion

- **Epochs completed**: `3000` / 3000 (`train_end_3000 = True`).
- **Final checkpoint**: `results/stage2_RQ-VAE/r4_10/rqvae_best.pth` (`sha256 = df28de1a78655fd32fe5d80a4bdb558f668d1d4744758dffe60bae7fbc8e9663`).
- **Final SIDs**: `results/stage2_RQ-VAE/r4_10/item_sids.json` (`sha256 = 9cd18ebb9eaa53906eb41004b63165cee2552b069df64d88060efc39866e660b`), `sids_for_hgrec.npy`.
- **Canonical reference SID SHA256**: `b08567c7fbaf06e585be37e4c2ceb379f6435e4d36e220c578ae3e6e10265d10`.

## 4. Frozen held-out Hard L1 ER and Stage 3 gate

Evaluated on `results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.npz` (52,855 positive transitions, 10 frozen matched negatives per row, seed 42), column 0 of the exported hard discrete SID only:

| Quantity | Value |
|:---|---:|
| `pos_share` | `0.0829628228` |
| `neg_share` | `0.0052048056` |
| **`ER_L1 = pos_share / neg_share`** | **`15.93965830607052`** |
| Canonical Resample `ER_L1` | `16.38953942232631` |
| `Δ ER_L1` vs canonical | `-0.44988111625579` |
| Strict `ER_L1 > 16.38953942232631` | `False` |
| **`stage3_allowed`** | **`False`** (FAIL — Stage 3 refused) |

## 5. Stage 3 disposition

Gate failed (`ER_L1` did not strictly exceed the canonical Resample baseline `16.38953942232631`). Per the issue contract, Stage 3 was **not** launched, queued, or prepared.
