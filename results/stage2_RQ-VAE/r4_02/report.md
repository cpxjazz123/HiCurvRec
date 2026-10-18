# R4_02 (Issue #52) — Symmetric row-column L1 InfoNCE on frozen Resample pairs

## 1. Sole mechanism and formula

- **Parent**: canonical Exp1a Resample (`stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE`, `EXPERIMENT_ARM = "resample"`).
- **Mechanism flag**: `R4_02_SYMMETRIC_L1`.
- **Formula**: `L1_new = 0.5 * CE_rows(S) + 0.5 * CE_rows(S.T)`.
- **Changed files vs parent**: `train_rqvae.py`.
- **Trainer SHA256**: `dc95ea1ff59391364fe351c236cdfa833cee41b9314666b9ef8806df6bcb7dfd`.

## 2. TDD RED → GREEN

Full verbatim RED (against the unchanged canonical Resample parent) and GREEN (against this arm's tree) logs, plus the single-delta diff, are recorded in [`tdd_red_green.md`](tdd_red_green.md). Runnable suite: `results/stage2_RQ-VAE/r4_02/test_r4_02.py` (`tests_passed = True`).

## 3. Stage 2 training completion

- **Epochs completed**: `3000` / 3000 (`train_end_3000 = True`).
- **Final checkpoint**: `results/stage2_RQ-VAE/r4_02/rqvae_best.pth` (`sha256 = 91078046358cc7566011dc5d66db365d6020b9839ddaca130c50cbd3d493c049`).
- **Final SIDs**: `results/stage2_RQ-VAE/r4_02/item_sids.json` (`sha256 = b97d0c952428c06e4704e3b3b2108840edb8fb536e4d3c9f916f3d75623e92d8`), `sids_for_hgrec.npy`.
- **Canonical reference SID SHA256**: `b08567c7fbaf06e585be37e4c2ceb379f6435e4d36e220c578ae3e6e10265d10`.

## 4. Frozen held-out Hard L1 ER and Stage 3 gate

Evaluated on `results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.npz` (52,855 positive transitions, 10 frozen matched negatives per row, seed 42), column 0 of the exported hard discrete SID only:

| Quantity | Value |
|:---|---:|
| `pos_share` | `0.0787437329` |
| `neg_share` | `0.0048945228` |
| **`ER_L1 = pos_share / neg_share`** | **`16.08813297255508`** |
| Canonical Resample `ER_L1` | `16.38953942232631` |
| `Δ ER_L1` vs canonical | `-0.30140644977123` |
| Strict `ER_L1 > 16.38953942232631` | `False` |
| **`stage3_allowed`** | **`False`** (FAIL — Stage 3 refused) |

## 5. Stage 3 disposition

Gate failed (`ER_L1` did not strictly exceed the canonical Resample baseline `16.38953942232631`). Per the issue contract, Stage 3 was **not** launched, queued, or prepared.
