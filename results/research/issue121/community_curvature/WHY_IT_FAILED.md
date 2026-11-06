# The community line, closed and optimised: what it produced, and what it cannot do

## What the community mechanism produced (a real result)

The standalone community learner is the one thing in this campaign that is a
clean, reproducible scientific result:

* the **mixed** encoder (a free 64-d branch and a hyperbolic 64-d branch, both
  learning from the same frozen 768-d embedding) separates the curvature-ordered
  item pairs **twice as well** as a capacity-matched Euclidean control
  (0.051717 against 0.039919 on the held-out ranking pairs, paired over 24 571
  anchors, permutation p = 0 of 20 000) — the hyperbolic geometry is doing the
  work, and the control is the same capacity without it;
* the **curvature-aware graph propagation** raises the hard AF3 gap of the
  partition from 124.52 to 134.52 (a plain GCN) to **145.11** (curvature
  weighted), and drops the curvature ranking loss by a factor of three
  (0.00898 → 0.00291): the neighbourhood is where the curvature signal lives;
* the **learnable curvature** — the item the plan deferred until the fixed
  curvature was validated — moves the community space to c = 1.1399 and adds
  **+0.00505** to the separation (0.056737 fixed → 0.061788 learned), the best
  separation any arm of the community experiment produced.

## What the community mechanism cannot do (measured, not argued)

The partition's information about the behaviour, on the same independent sample
the quality gate uses, with the gate's own yardstick (the ratio of the
positives' share to the negatives' share; higher is better):

| grouping | held-out | train pairs |
|---|---|---|
| **the resample arm's level-1 codes** | **16.03** | 23.42 |
| **the frozen baseline's level-1 codes** | **6.16** | 7.12 |
| the learnable-curvature partition (F) | 2.58 | 3.47 |
| the plain-GCN partition (D) | 2.77 | 3.59 |
| the curvature-aware partition (E) | 2.47 | 3.39 |
| a k-means partition of the transition profile (64 / 256 / 1024) | 1.53 / 1.43 / 1.33 | 2.11 / 1.56 / 1.46 |

**Every partition of the behaviour graph carries less than half the behaviour
information the identifiers already carry** — and the gap is there on the
training side as well (3.47 against 7.12), where the partition was fitted. A
community partition is therefore a *strictly weaker re-encoding* of what the
level-1 codes already hold, and the only thing a term pulling the codes
towards it can do is dilute them.

That is what the campaign's six arms measured, each from a different side:

| arm | the codes' ratio | recall |
|---|---|---|
| frozen baseline | 6.16 | 0.05554 |
| community-l1-rq (attraction) | 5.49 | 0.05263 |
| community-repulse-l1-rq (repulsion) | 5.23 | 0.05423 |
| community-pairs-guarded-rq (contrast + the repository's guard) | 3.03 | 0.04328 |
| *resample reference (the campaign's best)* | *16.03* | *0.06374* |

The community arms all moved the codes' ratio **down** and the recall **down**
with it; the arm that reached 0.06374 moved it **up** by a factor of 2.6. Read
correctly — the gate's error rate is a ratio, so a higher rate is better
discrimination — the campaign's ordering is monotone, and it says the recall
follows the identifiers' behaviour discrimination.

## The gate this line now has

Any future use of a community partition in the RQ-VAE has to clear one number,
measurable without a GPU: **the partition's held-out behaviour discrimination
must exceed the baseline codes' 6.16** (the resample arm's 16.03 is the bar to
beat). The optimised partition reaches 2.58. It is not close, and no arm of
this campaign cleared it.

## What is left for the campaign, and it is not the community

The lever the campaign's own table points at is the identifiers' behaviour
discrimination, and the mechanism that moved it is the **behaviour-pair
contrastive term on the residuals**: the arm that reached 0.06374 trains the
encoder on the resampled behaviour pairs and leaves the quantizer's assignment
alone, which is why its codes discriminate the held-out behaviour pairs 2.6
times better than the baseline's without touching prefix diversity.

The next arm therefore takes the machinery of the guarded community-pair arm —
the per-step pair sampling, the false-negative masking, the measured gradient
bound, the repository's anti-collapse guard — and changes one thing: the
positives are the training split's **transitions** rather than the community's
members, and the kernel is the contract's Poincaré distance rather than the
Euclidean one. Its prediction is falsifiable and it is the first arm in the
campaign with a positive one: the codes' level-1 error rate should rise above
6.47 and the recall should rise above 0.05554.

**Files.** `tools/research/science/community_curvature/` — the learner
(`model.py`, with the learnable curvature), the data (`data.py`), the metrics
(`metrics.py`), the runner (`run_experiment.py`), the two cheap measurements
that decide the line's fate (`alignment.py`, `partition_capacity.py`), and the
reports in `results/research/issue121/community_curvature/`.
