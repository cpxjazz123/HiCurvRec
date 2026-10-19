# R6 arm r6_02 (issue #74) — Stage 2 completed, ER gate FAILED

## Mechanism

`Half-strength residual STE: q_mix = stopgrad(q) + 0.5*(q-stopgrad(q))`

Exactly one treatment, inside `_residuals_with_trace` only. `RQVAE.forward`, the production VQ
layers and the hard SID export are untouched.

Distinguishing claim verified by a standalone RED→GREEN test: L2/L3 encoder gradients restored with decaying coefficients 0.5 and 0.25

## CGV (six gates, all PASS)

`spec_sha256 = 1a79ab1d73e98e7498d029d119690b817c113e4ea3b305e35a5b950f67fceb59`

| gate | result |
|:--|:--|
| G1 forward contract | 10 invariants; hard forward bitwise identical to the untouched control walk, 0 SID differences |
| G2 gradient propagation | L1/L2/L3 encoder paths non-zero where the untreated model has exactly zero |
| G3 gradient mathematics | deviation vs the independent oracle 1e-7 class; rule `half_ste_residual` |
| G4 parameter update | production AdamW lr 1e-3, weight_decay 1e-4, clip 1 |
| G5 distributed consistency | real two-rank, rank batches 24/20, pooled deviation 0.000e+00 |
| G6 real training entry | mechanism called, optimizer stepped, target encoder parameters moved |

CGV re-ran all six gates inside the `stage2_start` call itself, immediately before dispatch.

## RED → GREEN (standalone test)

`results/stage2_RQ-VAE/r6_02/test_r6_02.py`, exit 0.

RED: on the untreated Exp1a trace the L2 and L3 encoder gradients are EXACTLY zero and the level
Jacobians measure [1, 0, 0]. GREEN: on this arm every fixed seed delivers non-zero L2/L3 encoder
gradients, the hard forward is unchanged, and a real AdamW step moves an encoder parameter.

## Stage 2 evidence

| item | value |
|:--|:--|
| train_end epoch (explicit, not defaulted) | 3000 |
| ckpt_saved epoch | 3000 |
| checkpoint | `rqvae_best.pth` |
| metrics rows | 3003 |
| final raw_unique | see `stage2_metrics.json` neighbours in the run log |
| SID sha256 | `25b4dbba2a8f9650a7b4133b5a4eb9f37f062258c8cbbf38a60408186e3860e5` |

## Hard-L1 ER gate — FAIL

| item | value |
|:--|:--|
| pos_share | 0.28941443572036707 |
| neg_share | 0.10013432977012582 |
| **Hard_L1_ER** | **2.8902618750708537** |
| baseline | 16.38953942232631 |
| delta | -13.499277547255454 |
| frozen pairs sha256 | `04d779208343a9e5cea769ddd5b600a4411c9313416a0fdba11ebaf0081125d0` |

`stage3_allowed = False`

Independent corroboration: the run's own final SID diversity collapsed relative to the r4_08 run
that historically scored ER 16.402628 (collision 0.556 / 0.707 / 0.850 against 0.0398). The
absolute positive match rate rose while the ratio to the matched negatives fell, which is the
signature of an L1 codebook losing discrimination, not of a measurement artefact. The evaluation
pipeline itself was validated by reproducing r4_08's recorded ER of 16.402628 through this same
script.

## Outcome

Restoring the L2/L3 behaviour gradients is real and verified, but on this configuration it costs
the L1 SID the discrimination the ER gate measures. The gate is the only Stage 2 quality criterion
the issue defines, and it fails by a wide margin.

**No Stage 3 was launched.** Per the issue: on FAIL, report ER and gate, close without Stage 3.

## Gitignored bytes (real paths and hashes)

| file | size | sha256 | tracked |
|:--|--:|:--|:--|
| `rqvae_best.pth` | 4.6 MB | `8427306b9a829d6aaa0b7a504c2894b2a90d90cbbdb15617d79e9711e9f26011` | yes |
| `sids_for_hgrec.npy` | 0.8 MB | `25b4dbba2a8f9650a7b4133b5a4eb9f37f062258c8cbbf38a60408186e3860e5` | yes |
| `item_sids.json` | 1.3 MB | `ef09df22e66a27425ce2ce7ebb964d2dc7d4ee356cf4553e22f04c51b788608f` | NO (gitignored) |

The checkpoint and the SID array are gitignored, so their bytes exist only at the paths and hashes above; they are not in the repository.

## Audit binding

This verdict is bound to the verification it depends on, not merely filed beside it:

| item | value |
|:--|:--|
| declaration on disk | `spec.json` |
| declaration sha256 (RECOMPUTED by the gate) | `1a79ab1d73e98e7498d029d119690b817c113e4ea3b305e35a5b950f67fceb59` |
| declaration sha256 (as covered by the CGV report) | `1a79ab1d73e98e7498d029d119690b817c113e4ea3b305e35a5b950f67fceb59` |
| the two agree | True |
| trainer sha256 the CGV report judged | `02f31155c173c39dde984e7886df7b6a9939e34fbc029b526ac8f550d885472c` |
| trainer sha256 now | unchanged, otherwise the gate records `cgv_stale` |
| CGV gates in that report | `{"G1": "passed", "G2": "passed", "G3": "passed", "G4": "passed", "G5": "passed", "G6": "passed"}` |
| CGV verified at | `2026-10-19T05:19:52Z` |

The gate RE-HASHES `spec.json` and refuses when it disagrees with the report, so neither a doctored
report nor an edited declaration can pass unnoticed. Both tamper modes were exercised: falsifying
the hash inside the report, and silently changing `update_contract.lr` in the declaration on disk.
Both produced `cgv.ok = false` and a `cgv_spec_mismatch` failure.
