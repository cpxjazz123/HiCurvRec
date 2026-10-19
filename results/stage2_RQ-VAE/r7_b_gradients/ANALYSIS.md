# Track B (issue #81) — what actually reaches the optimizer

Observational only. No branch, no restore, no counterfactual, and therefore no dependence on
issue #79 or on causal validity. Every number comes from the frozen step track A captured, rebuilt
once in a private branch, with each loss term's gradient taken separately by autograd.

## The question

Why is healthy R4-08's weighted behaviour-to-base encoder gradient ratio 314.7x, larger than the
failed R6 arms' 146.5x-167.0x? If bigger is healthier, magnitude cannot explain the collapse, and
the question becomes whether the **applied** direction differs.

## Measurement

At step 160, per arm, on the encoder parameters only:

| arm | base encoder grad L2 | weighted deep encoder grad L2 | weighted/base ratio | cos(base, weighted deep) |
|:--|--:|--:|--:|--:|
| r4_08 (healthy) | 2.0921e-04 | 5.6015e-02 | **267.7** | -0.2573 |
| r6_02 (failed) | 4.1394e-04 | 9.4911e-02 | **229.3** | -0.2170 |
| r6_03 (failed) | 5.1811e-04 | 1.0723e-01 | **207.0** | -0.1924 |
| r6_04 (failed) | 6.0495e-04 | 1.8239e-01 | **301.5** | -0.0437 |

**Issue #78's ordering does not reproduce.** #78 reported R4-08 at 314.7x and the R6 arms at
146.5x-167.0x, i.e. the healthy arm clearly on top. Here the healthy arm is 267.7x while the failed
r6_04 is 301.5x — the *failed* arm has the larger ratio. Track A measured the run-to-run spread on
this quantity's neighbours at ±5 to ±6 L1 codes and ±0.06 tuple collision, so a ratio ordering of
267.7 against 301.5 is not a separation this protocol can support. The one robust observation is
that the deep gradient **opposes** the base gradient on every arm (cosine between -0.04 and -0.26),
which is the conflict R4-08's PCGrad exists to resolve.

## The decisive control: the realised update barely follows this step's gradient

| arm | cos(update, this step's gradient) | cos(update, Adam direction m/sqrt(v)) | expected |
|:--|--:|--:|:--|
| r4_08 | -0.2042 | **-0.9334** | negative |
| r6_02 | -0.2168 | **-0.9463** | negative |
| r6_03 | -0.2188 | **-0.9500** | negative |
| r6_04 | -0.2021 | **-0.9406** | negative |

AdamW updates `p -= lr * m / sqrt(v)`, so the realised step must be **anti-parallel** to that
direction, and the measured cosines of -0.93 to -0.95 confirm the realised step *is* the AdamW step.
Because `m` and `v` are exponential averages over the training history rather than this step's
gradient, the step tracks the history and not the step.

Quantified: the current step's gradient accounts for about **4%** of the update's energy
(0.20^2), while the moment history accounts for about **88%** (0.94^2).

This is the answer to the lane's question, and it is a negative one for the framing that motivated
it. **Comparing per-step gradient magnitudes across arms — the 314.7x versus 146.5x comparison —
measures a component that contributes roughly a twentieth of the realised update.** A healthy arm
may indeed have a larger raw gradient and still make a less disruptive optimizer move; but so may a
failed arm, and the quantity does not order the cohort.

## Two controls that earned their keep

1. **The Adam-direction control caught a bug in the measurement itself.** The first version built the
   update vector by iterating `sorted(state_dict)` while the gradient vectors iterated
   `named_parameters()`. The two orders differ, so the cosine was over a permuted vector and read
   ≈0. The control was added precisely to distinguish "the update is orthogonal to the gradient"
   from "these two vectors disagree in order", and it did: with the order fixed the cosine moved
   from ≈0.001 to -0.20, and the Adam alignment appeared at -0.94.
2. **A competing explanation was tested and rejected.** The first hypothesis was that Adam's
   per-coordinate normalisation drives `m/sqrt(v)` towards `sign(g)`, which would make the
   orthogonality a design property. Measured `cos(update, sign(g))` was ≈0, so that explanation
   was wrong and was dropped rather than reported.

Weight decay was also ruled out: at step 160 the decoupled weight-decay move is 4.6e-07 against a
total update of 3.7e-02, a share of 0.00001.

## Limits

One frozen step per arm, on a 32+128 stress schedule that differs from the production epoch-1500
ramp. Observational: the cosines say what the optimizer did, not what caused a SID to collapse.
The 314.7x figure itself is not reproduced, and this lane does not confirm it.

## Next decision

Stop treating the weighted gradient ratio as a health signal. Any early-risk indicator should be
built on quantities the optimizer actually uses — the moment statistics, the realised update
direction, or the codebook occupancy that track C measures — and every such quantity has to be read
against track A's run-to-run noise floor.
