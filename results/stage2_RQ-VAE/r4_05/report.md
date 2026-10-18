# R4_05 (Issue #55) — One-skip-history anchor in unchanged three-level Resample contrastive supervision

## 1. Sole mechanism and formula

- **Parent**: canonical Exp1a Resample (`stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE`, `EXPERIMENT_ARM = "resample"`).
- **Mechanism flag**: `R4_05_LAG2_SOURCE`.
- **Formula**: `source = history[-2] if len(history) >= 2 else history[-1]; target unchanged`.
- **Changed files vs parent**: `train_rqvae.py`.
- **Trainer SHA256**: `db9fef7303252c279fbaa278e283b943d006e203c2b09e1fcf195aca9b4f9634`.

## 2. TDD RED → GREEN

Full verbatim RED (against the unchanged canonical Resample parent) and GREEN (against this arm's tree) logs, plus the single-delta diff, are recorded in [`tdd_red_green.md`](tdd_red_green.md). Runnable suite: `results/stage2_RQ-VAE/r4_05/test_r4_05.py` (`tests_passed = True`).

## 3. Stage 2 training completion

- **Epochs completed**: `3000` / 3000 (`train_end_3000 = True`).
- **Final checkpoint**: `results/stage2_RQ-VAE/r4_05/rqvae_best.pth` (`sha256 = d1f755cf57889720933a20da5d73c584ce06298daa556ff6d4080de2d614bc56`).
- **Final SIDs**: `results/stage2_RQ-VAE/r4_05/item_sids.json` (`sha256 = 25e63388abf8e5fb8492f029c6f48452f9f4bb4994413b83545108803b94cf44`), `sids_for_hgrec.npy`.
- **Canonical reference SID SHA256**: `b08567c7fbaf06e585be37e4c2ceb379f6435e4d36e220c578ae3e6e10265d10`.

## 4. Frozen held-out Hard L1 ER and Stage 3 gate

Evaluated on `results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.npz` (52,855 positive transitions, 10 frozen matched negatives per row, seed 42), column 0 of the exported hard discrete SID only:

| Quantity | Value |
|:---|---:|
| `pos_share` | `0.0774571942` |
| `neg_share` | `0.0051139911` |
| **`ER_L1 = pos_share / neg_share`** | **`15.14613392526822`** |
| Canonical Resample `ER_L1` | `16.38953942232631` |
| `Δ ER_L1` vs canonical | `-1.24340549705809` |
| Strict `ER_L1 > 16.38953942232631` | `False` |
| **`stage3_allowed`** | **`False`** (FAIL — Stage 3 refused) |

## 5. Stage 3 disposition

Gate failed (`ER_L1` did not strictly exceed the canonical Resample baseline `16.38953942232631`). Per the issue contract, Stage 3 was **not** launched, queued, or prepared.
