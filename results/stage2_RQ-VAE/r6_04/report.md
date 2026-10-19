# R6 arm r6_04 (issue #76) — Stage 2 completed, ER gate FAILED

## Mechanism

`Hard forward / soft codeword backward surrogate over the raw codebook`

Exactly one treatment, inside `_residuals_with_trace` only. `RQVAE.forward`, the production VQ
layers and the hard SID export are untouched.

Distinguishing claim verified by a standalone RED→GREEN test: behaviour path newly reaches the CODEXBOOKS; deep alignments 0.942/0.903

## CGV (six gates, all PASS)

`spec_sha256 = ba72757be6c89de715a3a85f2c12fc473f99f6dfaff3173e1b1fa60a6ca63686`

| gate | result |
|:--|:--|
| G1 forward contract | 10 invariants; hard forward bitwise identical to the untouched control walk, 0 SID differences |
| G2 gradient propagation | L1/L2/L3 encoder paths non-zero where the untreated model has exactly zero |
| G3 gradient mathematics | deviation vs the independent oracle 1e-7 class; rule `soft_codeword_surrogate` |
| G4 parameter update | production AdamW lr 1e-3, weight_decay 1e-4, clip 1 |
| G5 distributed consistency | real two-rank, rank batches 24/20, pooled deviation 0.000e+00 |
| G6 real training entry | mechanism called, optimizer stepped, target encoder parameters moved |

CGV re-ran all six gates inside the `stage2_start` call itself, immediately before dispatch.

## RED → GREEN (standalone test)

`results/stage2_RQ-VAE/r6_04/test_r6_04.py`, exit 0.

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
| SID sha256 | `8d39e8a7b4484a651a0362ca8149c16558e16c1d05612f801ef2751d98ac900f` |

## Hard-L1 ER gate — FAIL

| item | value |
|:--|:--|
| pos_share | 0.46816762841736825 |
| neg_share | 0.3036760949768234 |
| **Hard_L1_ER** | **1.5416677050365089** |
| baseline | 16.38953942232631 |
| delta | -14.847871717289799 |
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
| `rqvae_best.pth` | 4.6 MB | `0fec712c8eaf5dc81ad4786ecc7ffbc4b6bb38860e8b0d0dcd6ad379496308bf` | yes |
| `sids_for_hgrec.npy` | 0.8 MB | `8d39e8a7b4484a651a0362ca8149c16558e16c1d05612f801ef2751d98ac900f` | yes |
| `item_sids.json` | 1.3 MB | `fc7f5ed4b074767863d8f35285cc00118eb83dfb9212b2738fdab839d8d31fab` | NO (gitignored) |

The checkpoint and the SID array are gitignored, so their bytes exist only at the paths and hashes above; they are not in the repository.
