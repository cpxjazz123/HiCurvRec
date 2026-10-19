# R6 arm r6_03 (issue #75) — Stage 2 completed, ER gate FAILED

## Mechanism

`Direct encoder latent gradient bridge on the CE-facing deep traces`

Exactly one treatment, inside `_residuals_with_trace` only. `RQVAE.forward`, the production VQ
layers and the hard SID export are untouched.

Distinguishing claim verified by a standalone RED→GREEN test: both deep levels reach the latent at 0.5; measured NOT R6-02's 0.25

## CGV (six gates, all PASS)

`spec_sha256 = f166ba02d373568cdb9cb2b6bfc839f7aadc3fd787957d901635f2069bfe4fe7`

| gate | result |
|:--|:--|
| G1 forward contract | 10 invariants; hard forward bitwise identical to the untouched control walk, 0 SID differences |
| G2 gradient propagation | L1/L2/L3 encoder paths non-zero where the untreated model has exactly zero |
| G3 gradient mathematics | deviation vs the independent oracle 1e-7 class; rule `direct_latent_bridge` |
| G4 parameter update | production AdamW lr 1e-3, weight_decay 1e-4, clip 1 |
| G5 distributed consistency | real two-rank, rank batches 24/20, pooled deviation 0.000e+00 |
| G6 real training entry | mechanism called, optimizer stepped, target encoder parameters moved |

CGV re-ran all six gates inside the `stage2_start` call itself, immediately before dispatch.

## RED → GREEN (standalone test)

`results/stage2_RQ-VAE/r6_03/test_r6_03.py`, exit 0.

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
| SID sha256 | `21730f48e316d06c0bca2e9a7fd59ae6b82fc66015d054f561d72b6adaae1ca4` |

## Hard-L1 ER gate — FAIL

| item | value |
|:--|:--|
| pos_share | 0.4268848737110964 |
| neg_share | 0.23411976161195724 |
| **Hard_L1_ER** | **1.8233611326609775** |
| baseline | 16.38953942232631 |
| delta | -14.56617828966533 |
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
| `rqvae_best.pth` | 4.6 MB | `076909b200678f6c35d8e946255623b178ed6b4a7c7f1095647f4626ac3be1cf` | yes |
| `sids_for_hgrec.npy` | 0.8 MB | `21730f48e316d06c0bca2e9a7fd59ae6b82fc66015d054f561d72b6adaae1ca4` | yes |
| `item_sids.json` | 1.2 MB | `31d5a937d30a7bef537c40b4c8678ca8462c8ae7881b166a5a8357be062431ea` | NO (gitignored) |

The checkpoint and the SID array are gitignored, so their bytes exist only at the paths and hashes above; they are not in the repository.
