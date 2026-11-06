# The behaviour contrast, three placements, one answer: it spends the code space

## What was tried, and what each placement did

The campaign's table says the recall follows the identifiers' behaviour
*discrimination* — the ratio of the held-out positives' code share to the
negatives' — and the arm that reached 0.06374 has the highest ratio measured
(16.03 against the baseline's 6.16). The community partition could not supply
that: it carries 2.4 to 2.8 times the behaviour information while the codes
carry 6.2 to 16.0, so pulling the codes towards it diluted them (five arms, one
outcome).

This round moved to the direct lever: contrast the encoder on the training
split's **transitions**, resampled every step, with the repository's own
anti-collapse guard and the measured gradient bound, and change only *where* in
the quantiser the contrast lands.

| arm | where the contrast lands | L1 Hard ER | L2 prefixes | L3 collision | L1 positive hit | verdict |
|---|---|---|---|---|---|---|
| frozen baseline | — | **6.47** | **7414** | **0.2745** | 0.0864 | — |
| behaviour-pairs-curved | the encoder's output (level-1's input) | 3.98 | 4110 | 0.4116 | **0.1701** | refused by the gate |
| behaviour-pairs-levels | the residuals of levels 2 and 3 | 3.93 | **3688** | **0.4490** | 0.1793 | refused by the gate |

Both placements fail, and they fail *the same way*: the coarse code becomes a
much better predictor of the behaviour partners (the positive hit rate 0.0864 →
0.1701 → 0.1793, roughly doubled), the behaviour negatives rise with it
(0.0134 → 0.0427 → 0.0457), the ratio falls (6.47 → 3.98 → 3.93), the level-2
prefixes fall (7414 → 4110 → 3688) and the three-level collision rises (0.2745
→ 0.4116 → 0.4490) until the owner's 35% gate refuses the arm.

Moving the contrast off the encoder's output did **not** protect the code space
— it made the collapse slightly worse, because the residuals the contrast then
shapes are the ones the fine codebooks quantise, so the fine codebooks lose
their diversity instead of the coarse one losing its discrimination.

## What this establishes

The failure is not about *where* the behaviour is imposed. It is about
**imposing** it: a contrastive term that pulls behaviour-related latents
together spends codebook capacity on that structure, and capacity spent on one
structure is capacity unavailable to the rest. The two ends of the campaign now
agree from opposite directions:

* the community arms imposed the *partition* on the codes and lost
  discrimination (6.16 → 5.49 → 5.23 → 3.03);
* the behaviour arms imposed the *transitions* on the representation and lost
  prefix diversity (7414 → 4110 → 3688 level-2 prefixes).

The arm that reached 0.06374 imposed neither: it has the highest
discrimination (16.03) and the most level-2 prefixes (16145) of anything
measured. Whatever it does, it is not what these two families do.

## The hypothesis this leaves

The campaign's data support one reading, and it is testable without another
arm: **the resample arm's contrast does not shape the encoder's output or the
fine residuals in a way that groups behaviour — it changes which pairs the
model is *exposed* to, epoch by epoch, while its level-1 code stays nearly
non-predictive (the held-out positive hit is 0.0794, the baseline's 0.0864) and
its discrimination is the highest measured.** A contrast that pulls the latents
together makes the code predictive; something in that arm makes the code
*discriminative without being predictive*, and the difference is the target of
the next measurement, not of another arm.

Concretely, the next step is to read the behaviour-sample tree's
`resample` arm against the two arms above on the one quantity that separates
them — the level-1 code's positive and negative hit rates on the same held-out
sample — and find which of its terms produces the discrimination without the
predictiveness. That is a reading of existing artifacts, not a training run.

## The two defects this round found

1. **The unlabelled-positive mask was a python loop** over every successor of
   every anchor: four epochs per second instead of fourteen. It is a sparse
   matrix slice now, and the test suite pins it.
2. **The lift scale was the 95th percentile of the latents' norms**, which
   pushes the lifted points onto the ball's boundary where the contract's
   distance saturates: with it the arm collapsed outright (collision 1.000,
   three unique identifiers). The median keeps the lifted points inside the
   ball. The behaviour-sample arm's own ramp — the term joins over the second
   half of the budget — was added at the same time, so the reconstruction trains
   the representation before the behaviour supervision joins it.

**Files.** `stage2_RQ-VAE/TIGER_behaviour-pairs-curved-rq/` and
`stage2_RQ-VAE/TIGER_behaviour-pairs-levels-rq/` (each with its tests, its
reference contract and its metrics stream), and the two records
`behaviour_pairs_curved_failed.json` and the layer records of both arms.
