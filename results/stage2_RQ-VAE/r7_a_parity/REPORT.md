# Track A (issue #80) — control parity repaired, and the environment is not bit-reproducible

## What was broken in #79

Issue #79 branched counterfactuals **from inside the running loop**: it ran measurement forwards
between `backward()` and `optimizer.step()`, captured and restored state around them, and advanced
the loop to take counterfactual steps. Its audited origin then failed to match an independent probe
(r6_02 at step 96: 117 L1 codes against the probe's 110), and by its own rule it reported
`inconclusive` with no causal claim.

## What this track does instead

Two changes, both structural rather than parametric:

1. **The capture only copies.** It arms at `optimizer.zero_grad` — the first call of a loop
   iteration, before the forward — reads the model, the AdamW moments, the scheduler, all four RNG
   streams, the *actual* item and pair minibatches with their labels, the applied deep weight and
   the learning rate, and arms nothing else. It runs no forward, touches no `.grad`, draws no
   random number and never changes train/eval mode.
2. **Branches live in fresh objects.** A branch is a newly constructed `RQVAE`, optimizer and
   optimizer state loaded from the frozen tensors. Nothing is advanced and then restored in the
   training run; the training run finishes as an ordinary training run.

Because the branch calls the arm's own `_residuals_with_trace`, `behavior_contrastive_loss`,
`_r4_08_split_behavior_loss`, `_r4_08_apply_projection`, `assemble_losses` and `AdamW.step`,
`FULL` is not "a loss that resembles the arm's" — it *is* the arm's loss.

## Control results

`FULL` reproduces the real loop's post-step parameters **bit for bit** on every arm at every active
fork step:

| arm | step 96 | step 160 | FULL-control max abs difference |
|:--|:--|:--|:--|
| r6_02 | `identical` | `identical` | `0.0` |
| r6_03 | `identical` | `identical` | `0.0` |
| r6_04 | `identical` | `identical` | `0.0` |
| r4_08 | `identical` | `identical` | `0.0` |

**Inert control (non-vacuous).** Step 32 sits outside the active window, so `NO-DEEP` removes a term
that was never applied and must reproduce `FULL` exactly. It does: `max_abs_diff = 0.0`, and the L1
switch count matches. `BASE-ONLY` is deliberately **not** part of this control — the base term is
live at every step, so removing it legitimately changes the update at every step.

## The finding that outranks all of the above

The non-perturbation control **failed**, and not because of the capture. Two runs of the
**unmodified trainer with no hooks at all**, same seed, same protocol, and with
`CUBLAS_WORKSPACE_CONFIG=:4096:8` pinned, reach different final parameters:

```
run 1  b5756ea53a63604f4da06c0de5ebe21e99e7755dd432cc103c22592dfee695cd
run 2  53a514f8f0bf28a94a1ee36de40279b56ffc773f508d7efd838569baea7f04bb
identical: False
```

The trainer sets `torch.use_deterministic_algorithms(True)`, but cuBLAS reports on every matmul that
this does **not** make its own reductions deterministic without a pinned workspace, and the pinned
workspace does not fix it here either.

Two consequences, and the second is the important one:

1. The control was **restructured into three runs** — one instrumented, two hook-free. Only if the
   two hook-free runs agree first can a difference be attributed to the capture. As it stands, the
   environment is not bit-reproducible, so the honest verdict is *not applicable: the environment
   is not bit-reproducible*, not *the capture perturbs the run*.
2. **This reframes issue #79.** It compared an audited trajectory against an independent probe as
   though two runs of the same code land in the same place. They do not. Its 117-versus-110 gap is
   not evidence that instrumentation perturbed the run; it is at least partly a measurement of the
   environment. That does not rescue #79 — its own rule still forbids the causal claim, because a
   comparison against a different run cannot be a counterfactual — but the stated *cause* of its
   failure was wrong, and the repair here is what finally separates the two.

`FULL` parity is a **within-run** statement: same process, same tensors, same selected cuBLAS
algorithm. It is exactly the guarantee a counterfactual branch needs, and it holds.

## The noise floor, in the units the earlier issues used

Two hook-free runs of `r6_02`, identical seed and protocol, measured on the fixed train-item
population at the same steps (`tools/r7_a_noise_floor.py`):

| step | L1 used codes | raw three-level tuple collision | L1 entropy (normalised) |
|:--|:--|:--|:--|
| 96, run 1 | 44 | 0.7042 | 0.5032 |
| 96, run 2 | 39 | 0.7689 | 0.5036 |
| **spread** | **±5** | **0.065** | 0.0004 |
| 160, run 1 | 33 | 0.6611 | 0.4923 |
| 160, run 2 | 39 | 0.7178 | 0.5110 |
| **spread** | **±6** | **0.057** | 0.0187 |

Two consequences, and they change what earlier issues are allowed to claim.

**Issue #79's divergence was inside the noise.** It measured an audited origin of 117 L1 codes
against a probe's 110 — a gap of 7. Run-to-run spread on the same arm at the same step is ±5 to ±6.
That gap is therefore **not evidence that instrumentation perturbed the run**; it is a measurement of
the environment. #79's refusal to make a causal claim still stands on its own rule, but its stated
diagnosis was wrong.

**Issue #78's fine ordering does not survive; its coarse separation does.** At step 160 the raw
three-level tuple collisions were `[0.4391, 0.7442, 0.7666, 0.9165]`, sorted:

| consecutive gap | size | in units of the measured noise |
|:--|:--|:--|
| r4_08 → r6_02 | 0.3051 | **4.7x — real** |
| r6_02 → r6_03 | 0.0224 | **0.3x — not distinguishable** |
| r6_03 → r6_04 | 0.1499 | **2.3x — real** |

So the healthy-versus-failed separation that motivated this whole programme is real and larger than
the noise, while the ordering *within* the failed cluster is not. Any threshold fitted on those four
numbers is fitting three points and a noise band.

## What was measured

Per arm and step: the L1 assignments before and after each single update, the switch counts, and
the parameter difference of `NO-DEEP` and `BASE-ONLY` against `FULL`. These are now **admissible**,
because `FULL` reproduces the real step. They are one-step, on a frozen stress schedule that
differs from the production epoch-1500 ramp, and they are not a prediction of 3000-epoch ER.

## Tests

`tests/test_r7_a_frozen_replay.py` — 19 checks, 0 failures. Each way a replay can silently lie is
broken on its own and must be caught: a nonzero FULL difference, a missing AdamW moment, a wrong
minibatch, a doubled update, storage shared with the frozen bundle, a changed trainer hash, a
capture taken in eval mode, and an unknown intervention name.

## Cost

Per arm: one instrumented capture, two hook-free captures, and three branch replays per fork step.
No 3000-epoch Stage2, no Stage3. The frozen held-out pairs were never read; the frozen final gate
(`Hard-L1 ER > 16.38953942232631`) is untouched.

## Next decision

Because `FULL` parity now holds, the `FULL`-versus-`NO-DEEP` contrasts are usable, and any future
mechanism comparison must reproduce `FULL` bit-exactly before its own numbers are read. The
cross-run noise floor must be reported next to every across-arm number, including the ones in
issues #78 and #79.
