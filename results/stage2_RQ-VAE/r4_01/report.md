# R4_01 (Issue #51) — Candidate-popularity logit correction in baseline L1 InfoNCE

## 1. Sole mechanism and formula

- **Parent**: canonical Exp1a Resample (`stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE`, `EXPERIMENT_ARM = "resample"`).
- **Mechanism flag**: `R4_01_Q_CORRECTION`.
- **Formula**: `logits'_ij = logits_ij - 0.5 * log q_train(j)`.
- **Changed files vs parent**: `train_rqvae.py`.
- **Trainer SHA256**: `1683f5473fd1cf9d69491740c11f4d4f7c5a4e0549e18c892123101d9f855db7`.

## 2. TDD RED → GREEN

Full verbatim RED (against the unchanged canonical Resample parent) and GREEN (against this arm's tree) logs, plus the single-delta diff, are recorded in [`tdd_red_green.md`](tdd_red_green.md). Runnable suite: `results/stage2_RQ-VAE/r4_01/test_r4_01.py` (`tests_passed = True`).

## 3. Stage 2 training completion

- **Epochs completed**: `3000` / 3000 (`train_end_3000 = True`).
- **Final checkpoint**: `results/stage2_RQ-VAE/r4_01/rqvae_best.pth` (`sha256 = 09b77d025eca2c2173571b2e01ff6a504fe1155b9635ebca5221fa3b5732fe17`).
- **Final SIDs**: `results/stage2_RQ-VAE/r4_01/item_sids.json` (`sha256 = 82373387932a5e5a11d0ce3011fa786abd36629490d9ba9d1eca3dae1d24d175`), `sids_for_hgrec.npy`.
- **Canonical reference SID SHA256**: `b08567c7fbaf06e585be37e4c2ceb379f6435e4d36e220c578ae3e6e10265d10`.

## 4. Frozen held-out Hard L1 ER and Stage 3 gate

Evaluated on `results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.npz` (52,855 positive transitions, 10 frozen matched negatives per row, seed 42), column 0 of the exported hard discrete SID only:

| Quantity | Value |
|:---|---:|
| `pos_share` | `0.0782329013` |
| `neg_share` | `0.0053731908` |
| **`ER_L1 = pos_share / neg_share`** | **`14.55985915492958`** |
| Canonical Resample `ER_L1` | `16.38953942232631` |
| `Δ ER_L1` vs canonical | `-1.82968026739673` |
| Strict `ER_L1 > 16.38953942232631` | `False` |
| **`stage3_allowed`** | **`False`** (FAIL — Stage 3 refused) |

## 5. Stage 3 disposition

Gate failed (`ER_L1` did not strictly exceed the canonical Resample baseline `16.38953942232631`). Per the issue contract, Stage 3 was **not** launched, queued, or prepared.
