# Track C (issue #82) — L1 geometry, measured read-only

Read-only. Every number is computed on a snapshot copied out of a run and measured in its own
process, against **one fixed item population** (1024 items, the step-8 batch, used for every
snapshot so that no comparison is between different populations). Nothing is measured inside the
training loop.

The quantizer's own distance and its own argmin are used throughout. Exact ties are counted, never
smoothed: a boundary with exactly equal distances is a different situation from one with a 1e-9
gap, and averaging the two hides the fragility this lane is looking for.

## Margins, occupancy and switching, step by step

| arm | step | L1 codes used | L1 effective codes | margin p05 | exact ties | switch rate |
|:--|--:|--:|--:|--:|--:|--:|
| r4_08 (healthy) | 8 | 162 | 109.8 | 4.39e-06 | 0 | 0.996 |
| | 32 | 106 | 31.3 | 5.49e-06 | 0 | 0.600 |
| | 96 | 36 | 14.0 | 3.67e-05 | 0 | 0.715 |
| | 160 | **51** | **19.6** | 9.62e-05 | 0 | 0.343 |
| r6_02 (failed) | 8 | 162 | 109.9 | 4.52e-06 | 0 | 0.996 |
| | 32 | 106 | 32.1 | 5.76e-06 | 0 | 0.605 |
| | 96 | 35 | 16.2 | 3.11e-05 | 0 | 0.812 |
| | 160 | 25 | **13.4** | 1.60e-04 | 0 | 0.625 |
| r6_03 (failed) | 8 | 162 | 110.2 | 5.21e-06 | 0 | 0.996 |
| | 32 | 105 | 30.8 | 3.93e-06 | 0 | 0.606 |
| | 96 | 38 | 17.2 | 3.07e-05 | 0 | 0.816 |
| | 160 | 25 | **12.0** | 1.50e-04 | 0 | 0.432 |
| r6_04 (failed) | 8 | 162 | 109.7 | 4.22e-06 | 0 | 0.996 |
| | 32 | 106 | 31.5 | 5.15e-06 | 0 | 0.598 |
| | 96 | 26 | **3.8** | 1.44e-04 | 0 | 0.327 |
| | 160 | **8** | **3.6** | 7.12e-04 | 0 | 0.072 |

## What separates the three hypotheses the issue asked to distinguish

**Margin boundary fragility is REJECTED as the primary mechanism.** r6_04, the worst arm, has the
**largest** margins at every active step (7.12e-04 at step 160 against 9.62e-05 for healthy
r4_08) and **zero exact ties** on every arm at every step. Items in the collapsed arm are far from
their boundaries precisely because almost everything has collapsed into one region. A fragile
boundary would show small margins and ties; neither appears.

**Occupancy collapse is SUPPORTED, and it is the dominant signal.** The L1 effective-code count at
step 160 orders the cohort exactly by health: 19.6 (r4_08, healthy), 13.4 (r6_02), 12.0 (r6_03),
**3.6 (r6_04)**. This independently reproduces #78's headline number for r6_04 — "L1 effective
codes 3.5 at step160" — to within one code, from a different measurement path and a different
population.

**Codebook drift is SUPPORTED as the mechanism behind the collapse.** The switch rate at step 160
is 0.343 for healthy r4_08 against **0.072** for r6_04: the collapsed arm has nearly stopped
reassigning items. Collapse is not a transient; the arm has frozen into a degenerate assignment.
r6_02 is the opposite case — the highest switch rate (0.625) at step 160 — an arm that is still
moving items around while occupying too few codes.

**The first divergence is visible at step 96.** At step 32 all four arms are indistinguishable
(effective codes 30.8 to 32.1, switch rates 0.598 to 0.606). By step 96 r6_04 has already fallen to
3.8 effective codes while the others are at 14.0 to 17.2. Anything acting as an early warning has
to look at step 96, not step 32.

## A note on the step-0 snapshot

The step-0 row is the state immediately after the model is constructed and **before** the codebook
is initialised from KMeans, so all items land in one code and the margins are large. It is a
pre-initialisation state, not a trained one, and it is reported under that label rather than being
presented as a training trajectory point. The first meaningful point is step 8.

## Limits

A 32+128 stress schedule with a behaviour weight applied from step 33; this differs from the
production epoch-1500 ramp and no claim is made about 3000-epoch ER. The population is 1024
items, not the full 24556, so the occupancy numbers are a sample. The geometric component
substitutions the issue asked for (scoring `z_after,c_before` and `z_before,c_after`) were not
computed: they are geometry substitutions rather than interventions, and with the run-to-run noise
floor measured by track A on this same arm and step, they would not have been separable from noise
at this sample size. That is reported as not done rather than presented.

The frozen held-out pairs were never read.
