# The magnitude of the grouping force is not the lever: a bounded contrast collapses too

**What was tried.** The previous arm put the community partition into a contrast
over the encoder's latents and the code space collapsed (three-level collision
0.42–0.58 against the 0.35 gate). The diagnosis was precise: the two terms were
equalised by their **loss values**, and the contrast's **gradient** over the
encoder is far larger than the reconstruction's at that scale, so it dominated
the latents. This arm kept the same mechanism and bounded it by the gradient:

    scale = min(1, ||grad(community term)|| / ||grad(reconstruction)||)
                      over the encoder's parameters, measured every step

The bound is stated once and needs no constant: the community supervision may
not push the encoder harder than the reconstruction it is added to. Four tests
pin it (one when the contrast is weaker, the ratio when it is stronger, one
when its gradient vanishes, and that reading the norm does not touch `.grad`).

**What happened.** The collapse shrank and stayed.

| run | epochs | collision median | collision min | collision last | unique SIDs |
|---|---|---|---|---|---|
| unbounded contrast | 1349 | 0.48 | 0.420 | 0.491 | 12 508 |
| **bounded contrast** | **721** | **0.438** | **0.380** | **0.486** | 12 638 |
| cost term at level 1 | 3000 | — | — | 0.206 (passes 0.35) | 24 587 |
| baseline family | 3000 | — | — | ~0.11–0.20 | ~20 000 |

The bound cut the worst of it (0.42–0.58 → 0.38–0.49), but the collision
oscillated between 0.40 and 0.46 for the six hundred epochs before the run was
stopped, with no downward trend, and the unique identifier count stayed near
12 600 out of 24 587. The gate is 0.35; nothing in the trajectory approaches
it. The run was stopped at epoch 721 rather than spending the remaining 2 279.

**What this establishes.** The magnitude of the grouping force is **not** the
lever — its **direction** is. Two different strengths were measured:

* equal loss values (the contrast dominating the gradient): collision 0.42–0.58;
* the gradient itself bounded to the reconstruction's: collision 0.38–0.49.

Both collapse. Pulling the latents of a community together is a grouping force
that a 256-codebook cannot absorb without reuse, because the quantiser sees
exactly the latents being pulled together. Prefix diversity is what the decoder
conditions on, and it is what dies.

## The community mechanism, in every shape measured

| shape | where the community acts | L3 collision | Stage-3 recall@10 |
|---|---|---|---|
| cost at level 1 | the assignment's cost matrix | 0.3259 (passes) | 0.05263 |
| standalone community model | outside the RQ-VAE | — | not run (mechanism only) |
| pair contrast on latents | the encoder's latents | 0.42–0.58 (refused) | not run |
| **bounded pair contrast** | the encoder's latents, gradient-bounded | **0.38–0.49 (refused)** | not run |
| *reference*: frozen baseline | — | 0.2745 | 0.05554 |
| *reference*: resample (best in campaign) | pair sampling, code space untouched | passes | **0.06374** |

The community signal itself is real and measurable — the cost-term arm is the
only arm in this campaign that improved the coarse level's positive-hit rate
(0.0864 → 0.0989) and it did so without collapsing the codebook (256/256 codes
used). But it did not move the recall, and every attempt to make the community
act more strongly on the representation lost prefix diversity instead.

**Where this leaves the direction.** Not in another strength or another
placement of the same grouping force: four shapes, three of them refused by the
gate and the one that passed without a recall gain. If the community
information is to be used at all, it must act where prefix diversity is not at
stake — as a diagnostic of the partition, or on the generation side as a
conditioning feature — and the campaign's only near-threshold arm (resample,
0.06374) changed *which pairs the encoder was contrasted on per epoch* while
leaving both the code space and the latents' global structure alone.

**Evidence.** `community_pairs_bounded_failed.json` (trajectory, gate, reason),
the arm's tree `stage2_RQ-VAE/TIGER_community-pairs-bounded-rq/` with eleven
tests and its reference contract, and the machine admission that let it run:
all ten checks passed (reference agreement values 0.0, gradients 2.2e-16,
gradcheck on both sides, coupling displacement 0.69).
