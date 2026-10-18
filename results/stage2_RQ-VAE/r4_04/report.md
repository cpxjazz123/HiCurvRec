# R4_04 (Issue #54) — Unique-transition-uniform sampling for baseline Resample behavior batches

## 1. Sole mechanism and formula

- **Parent**: canonical Exp1a Resample (`stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE`, `EXPERIMENT_ARM = "resample"`).
- **Mechanism flag**: `R4_04_UNIQUE_UNIFORM`.
- **Formula**: `IID uniform draws with replacement from sorted unique train (history[-1], target) edges`.
- **Changed files vs parent**: `train_rqvae.py`.
- **Trainer SHA256**: `fdf31fc73c09eb97a3cc20e0fc617ed582b3b755782ddf6d55103dc23ee04f28`.

## 2. TDD RED → GREEN

Full verbatim RED (against the unchanged canonical Resample parent) and GREEN (against this arm's tree) logs, plus the single-delta diff, are recorded in [`tdd_red_green.md`](tdd_red_green.md). Runnable suite: `results/stage2_RQ-VAE/r4_04/test_r4_04.py` (`tests_passed = True`).

## 3. Stage 2 training completion

- **Epochs completed**: `3000` / 3000 (`train_end_3000 = True`).
- **Final checkpoint**: `results/stage2_RQ-VAE/r4_04/rqvae_best.pth` (`sha256 = a068c657790e5133983cc7f676abbe69bc4205dd6a57ec5aa73e326b204cf187`).
- **Final SIDs**: `results/stage2_RQ-VAE/r4_04/item_sids.json` (`sha256 = fa8afd97391a8c2844362410e937e8959bcdb704206b976514d1487630655b37`), `sids_for_hgrec.npy`.
- **Canonical reference SID SHA256**: `b08567c7fbaf06e585be37e4c2ceb379f6435e4d36e220c578ae3e6e10265d10`.

## 4. Frozen held-out Hard L1 ER and Stage 3 gate

Evaluated on `results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.npz` (52,855 positive transitions, 10 frozen matched negatives per row, seed 42), column 0 of the exported hard discrete SID only:

| Quantity | Value |
|:---|---:|
| `pos_share` | `0.0780437045` |
| `neg_share` | `0.0050004730` |
| **`ER_L1 = pos_share / neg_share`** | **`15.60726447219069`** |
| Canonical Resample `ER_L1` | `16.38953942232631` |
| `Δ ER_L1` vs canonical | `-0.78227495013562` |
| Strict `ER_L1 > 16.38953942232631` | `False` |
| **`stage3_allowed`** | **`False`** (FAIL — Stage 3 refused) |

## 5. Stage 3 disposition

Gate failed (`ER_L1` did not strictly exceed the canonical Resample baseline `16.38953942232631`). Per the issue contract, Stage 3 was **not** launched, queued, or prepared.
