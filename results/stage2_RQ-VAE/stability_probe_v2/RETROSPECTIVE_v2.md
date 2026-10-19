# RETROSPECTIVE v2: corrected measurement and an exact 32+128 schedule

Measurement version 2. Version 1 divided each level's gradient norm by the TOTAL pre-clip
norm of every parameter and every loss -- not by `||grad(base_loss, encoder)||` -- and
regenerated contrastive labels with a cyclic shift instead of observing the trainer's own
terms. Its per-level ratios and its `deep_gradient_domination` indicator were wrong, and its
step schedule could reach only 112 active steps out of a promised 128. Those reports remain
on disk under `stability_probe/` and are not claimed to be correct.

Protocol, identical for all four arms: `seed=42`, cold start, 32 baseline steps then 128
steps at behaviour weight 0.20, snapshots at 0/8/16/32/64/96/128/160, single process, and the
arm's real loaders, loss assembly, optimizer, clipping and SID extraction. The trainer
exposes a probe-only per-step weight hook, proven bit-identical in production by an A/B run
against the pre-hook source on the same seed.

| arm | outcome | final Hard-L1 ER | status | risk | steps | active | mechanism calls | schedule agrees |
|:--|:--|--:|:--|:--|--:|--:|--:|:--|
| `r4_08` | healthy | 16.402628 | valid | high | 160 | 128 | 384 | True |
| `r6_02` | collapsed | 2.8902618750708537 | valid | high | 160 | 128 | 384 | True |
| `r6_03` | collapsed | 1.8233611326609775 | valid | high | 160 | 128 | 384 | True |
| `r6_04` | collapsed | 1.5416677050365089 | valid | high | 160 | 128 | 384 | True |

## Corrected trajectories at step 160

| arm | L1 used | L1 eff.codes | L1 collision | 3-level tuple collision | neg same-code | ||g_base_enc|| | max weighted ratio | min cosine vs base |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `r4_08` | 116 | 20.5 | 0.9953 | 0.4391 | 0.0703 | 1.916e-04 | 314.7 | -0.3396 |
| `r6_02` | 65 | 15.4 | 0.9974 | 0.7442 | 0.0892 | 4.960e-04 | 146.5 | -0.4972 |
| `r6_03` | 72 | 13.3 | 0.9971 | 0.7666 | 0.1046 | 5.025e-04 | 167.0 | -0.5832 |
| `r6_04` | 54 | 3.5 | 0.9978 | 0.9165 | 0.3531 | 5.796e-04 | 156.9 | -0.5374 |

## Do the early indicators still separate the arms?

In final-ER order ['r4_08', 'r6_02', 'r6_03', 'r6_04']:

- three-level tuple collision: [0.4391, 0.7442, 0.7666, 0.9165] -> monotone: **True**
- train-only negative same-code rate: [0.0703, 0.0892, 0.1046, 0.3531] -> monotone: **True**

At least one continuous early indicator still ranks the four arms in the same order as the
final ER after the correction. That is an OBSERVATION on one healthy arm and three
failures, not a calibrated predictor: no threshold, no sensitivity, no false-positive
rate and no GPU-saving guarantee is claimed, and none may be derived from these four arms.

## Gradient attribution, corrected

The denominator is `||grad(base_loss, encoder_parameters)||` from the same optimizer step, and
the per-level terms are the trainer's OWN contrastive losses, observed as it built them: the
real pair minibatch, the real source/target ids, the real duplicate and false-negative mask,
the real temperature. A ratio is reported as `undefined` with a reason when either norm is
exactly zero or a level is disconnected from the encoder, never as 0 and never as a large
fabricated number. `preclip_total_norm` is still reported, separately, and is never the
denominator.

`risk_calibration` remains **uncalibrated**: these four arms are not a calibration plus
independent evaluation split.
