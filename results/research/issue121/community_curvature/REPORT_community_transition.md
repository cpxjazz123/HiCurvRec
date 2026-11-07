# The community transition carries what the identifiers do not

The campaign so far asked whether the next item is in the **same** community as
the current one, and measured that at the chance level. That is the wrong
question for a generative recommender: a user who buys a shampoo may next buy a
conditioner — a different community and a perfectly stable relation. The
question that matters is which community comes next.

This report records three things: a correction to the previous report, the
measurement of the community transition, and the one measurement that decides
whether the community is worth carrying into the generator.

## 1. The correction

The previous report claimed the curvature-aware propagation lifts the hard AF3
gap from 124.52 (no graph) through 134.52 (a plain GCN) to **145.11**, and
treated the ordering of the arms as the result of the mechanism.

A reproducible re-run of the whole arm table (the same seed, the same code, the
deterministic cuBLAS workspace) gives a different ordering:

| arm | separation (valid) | AF3 gap (valid) | community transitivity | curvature |
|---|---|---|---|---|
| A — free branch, 128 communities | 0.0399 | 118.10 | 0.0687 | fixed |
| B — mixed geometry, 64 + 64 | 0.0517 | 124.52 | 0.0661 | fixed |
| C — B plus the neighbour terms | 0.0599 | 109.81 | 0.0720 | fixed |
| D — B plus a plain GCN propagation | **0.0665** | **139.47** | 0.0677 | fixed |
| E — B plus the curvature-aware propagation | 0.0639 | 134.54 | 0.0652 | fixed |
| F — E plus a learnable curvature | 0.0526 | 133.64 | 0.0655 | 1.1420 |
| G — F with 1024 communities | 0.0116 | 161.08 | 0.0730 | 1.0651 |
| H — F plus the curvature-consistency head | 0.0637 | 140.92 | 0.0665 | 1.1005 |
| I — F plus the cross-space consistency term | 0.0593 | 135.15 | 0.0672 | 1.1001 |

The curvature-aware propagation does **not** beat the plain GCN on the AF3 gap
here (134.54 against 139.47), and the earlier 145.11 was a property of an
earlier version of the shared model, not of the mechanism. The arm table of a
single run remains a valid comparison — every arm is seeded the same way and
the run reproduces bit for bit — but the cross-run spread that was reported as
"GPU non-determinism" was the shared model's code changing between runs. The
cuBLAS workspace is now pinned anyway (`CUBLAS_WORKSPACE_CONFIG=:4096:8` in the
Stage-2 launch environment, with a test), because the warning torch prints is
real and the guard is free.

Two consequences for the campaign's story:

* the **mixed geometry** is the one robust modelling gain (A → B: +30% on the
  separation, and the propagation arms another +25% over B);
* the **AF3 curvature does not earn its place inside the community model**. The
  edge-level statistic still separates communities from codes (the ledger's
  robust finding), but the learned partition is no better for having it.

## 2. The community transition

Everything below is measured on the **validation** split (57 439 held-out
interactions), with the transition matrix estimated on the **train** split
(339 519 sequences), and with the community assignment of arm E.

| predictor of the next community | NLL | hit@1 | hit@5 | hit@10 |
|---|---|---|---|---|
| global community frequencies | 4.1531 | 0.0196 | 0.1061 | 0.1878 |
| **community transition** (current community → next community) | **4.0440** | **0.0459** | **0.1775** | **0.3029** |
| community transition, **targets shuffled** (the control) | 4.2663 | 0.0156 | 0.0879 | 0.1739 |
| community transition along the whole prefix | 5.6058 | **0.0548** | **0.2041** | **0.3422** |
| the old hypothesis (the next item is in the current community) | rate 0.0452 against a modal-community base rate of 0.0158 | | | |
| SID alone (the frozen baseline's codes, memorisation estimate) | 19.4011 | 0.0464 | 0.1157 | 0.1945 |
| SID alone (the resample arm's codes) | 19.7284 | 0.0455 | 0.1061 | 0.1844 |
| **transition conditioned on the SID as well** | 11.9671 | 0.0522 | 0.1789 | 0.2847 |

Three readings, in order of importance:

1. **The transition is real and it is not popularity.** It beats the global
   frequencies on every metric (hit@1 2.3×, hit@10 +61%) and the shuffled
   control is *worse* than the global baseline, which is what a destroyed
   transition structure must look like. The signal is the relation between
   communities, not their sizes.
2. **The community carries information the identifiers do not.** The
   identifiers' own community-level prediction reaches hit@10 0.1945; the
   transition reaches 0.3422, a factor of 1.76. Conditioning the transition
   table on the identifiers as well does **not** improve it (0.2847, and a much
   worse likelihood), so the identifiers are not a substitute for the
   community structure.
3. **The earlier "chance level" reading was a measurement artefact.** The
   campaign compared the target against the *modal community of the whole
   prefix*; against the *last item's* community the rate is 0.0452, 2.9× the
   base rate. The prefix is a weaker statistic than its last step.

The prefix chain is the sharpest predictor (hit@10 0.3422) and the least
calibrated (its likelihood is worse than the single-step transition, because
the product over a long prefix over-penalises). The single-step transition is
the better-behaved estimator.

## 3. The measurement that decides it — and the first negative

The community's information is at the *community* level. Whether it helps pick
an **item** is a separate question, and the first answer is negative: as a soft
prior on top of a popularity ranking over the 1 000 most popular items, the
community transition only hurts.

| λ (the prior's weight) | 0.00 | 0.25 | 0.50 | 1.00 | 2.00 | 4.00 |
|---|---|---|---|---|---|---|
| validation recall@10 | **0.0279** | 0.0241 | 0.0198 | 0.0151 | 0.0130 | 0.0119 |

The λ is fitted on the train split and the fit chose λ = 0. The reason is
visible in the numbers above: the top-1 community of the transition covers
only 5.5% of the next items, and inside a community the prior is nearly
uniform, so tilting the ranking towards it throws away popularity evidence
without adding discrimination.

This is a statement about a **popularity baseline**, which is a weak
recommender, not about the generator. The question that decides the mechanism
is whether the prior adds anything to a model that already ranks items well.
That is measured next, against the frozen Stage-3 model on the validation
split: the model's own beams are dumped with their scores, the community prior
is applied to them, and the re-ranking is compared with the model's ordering.

## 4. What follows

* the community's **model** side (AF3 curvature inside the partition, the
  learnable curvature, the consistency head, the cross-space term) contributes
  nothing beyond the mixed geometry, and the cross-space term has nothing to
  align — the two branches already produce nearly identical assignments;
* the community's **transition** side is where the information is, and it is
  information the identifiers do not contain;
* so the community should enter as a **condition on the generator's decoding**,
  not as a constraint on the level-1 code — which is the reframe this campaign
  started from.

## Files

* `tools/research/science/community_curvature/transition_information.py` — the
  transition measurement and the SID comparison;
* `tools/research/science/community_curvature/item_level_prior.py` — the
  item-level prior measurement (the negative one);
* `results/research/issue121/community_curvature/community_transition_information.json` —
  the numbers above;
* `results/research/issue121/community_curvature/item_level_prior.json` — the
  numbers in section 3;
* `tools/research/stage2/contract.py` — the deterministic launch environment,
  with `tools/research/stage2/tests/test_launch_determinism.py`.
