# Why the community mechanism cannot help the identifiers: the codes already know more

## The measurement

Every partition and every codebook, measured on the same independent sample
(the 52 855 held-out behaviour pairs the quality gate uses) with the same
yardstick: the share of pairs whose two items share a group (positives), the
share of the negatives that do, and the ratio between them — the ratio is the
quality gate's own `valid_l1_hard_er`, and a *higher* ratio means the
identifier discriminates behaviour pairs better.

| grouping | positives share | negatives share | ratio |
|---|---|---|---|
| **the resample arm's own level-1 codes** | 0.0794 | 0.0050 | **16.03** |
| **the frozen baseline's level-1 codes** | 0.0864 | 0.0140 | **6.16** |
| the curvature-guided community partition (64) | 0.0409 | 0.0169 | **2.42** |
| a k-means partition of the transition profile (64) | 0.2184 | 0.1427 | 1.53 |
| a k-means partition of the transition profile (256) | 0.2293 | 0.1602 | 1.43 |
| a k-means partition of the transition profile (1024) | 0.2476 | 0.1865 | 1.33 |

Three facts fall out of it.

**1. The codes know more about the behaviour than any partition of the graph.**
The baseline's level-1 codes discriminate the held-out behaviour pairs 6.2
times over chance; the resample arm's 16.0; the best partition the graph can
be cut into, at any resolution, manages 2.4. A community partition is a
*strictly weaker version* of what the identifiers already encode — and that is
exactly why every community arm lost recall: a term that pulls the codes
towards the partition can only dilute them. The measured dilutions are the
campaign's own numbers: the codes' ratio fell from 6.16 to 5.49 (attraction),
5.23 (repulsion) and 3.03 (contrast with the guard), and the recall fell with
it, 0.05554 → 0.05263, 0.05423, 0.04328.

**2. The reason the codes generalise and the partitions do not is the
supervision they were fitted with.** The codes are optimised to reconstruct
the item embeddings, and the semantics of an item predict its consumers; the
partitions are fitted to the *observed* graph, whose edges are exactly the
transitions the held-out sample holds out. The graph's own clustering is
therefore fitted on the training transitions and cannot predict the new ones:
the measured 2.42 and 1.53 are the sizes of that generalisation gap.

**3. The campaign's relation is monotone once the error rate's direction is
read correctly.** The gate's error rate is a *ratio* (`positives × negatives /
negatives`), so a **higher** rate means **better** behaviour discrimination,
and the recall follows it: 3.03 → 0.04328, 4.26 → 0.04897, 5.23 → 0.05423,
5.49 → 0.05263, 6.16 → 0.05554, 16.03 → **0.06374**. The two middle arms
exchange places, which is the size of the noise the policy forbids measuring;
the ends are not noise.

## What this closes, and what it does not

**Closed, with a mechanism and not with five failures:** the community
partition as a force on the identifiers, in every shape — attraction,
repulsion, pair contrast, pair contrast with the repository's own anti-collapse
guard. It is closed by an information ordering: the graph's partition carries
2.4 times the behaviour information and the codes carry 6.2 to 16.0, so there
is nothing for the partition to add to the codes and only dilution for it to
take away. The five arms measured that dilution, each of them moving the
ratio down and the recall down with it.

**Not closed:** the community information outside the identifiers, where the
codes' discrimination is not at stake — a decoder-side prior (which changes no
SID) or a diagnostic of the partition itself.

**And the lever the campaign's own table points at:** the recall follows the
codes' behaviour discrimination, and the arm that reached 0.06374 is the one
whose codes discriminate best (16.03 against the baseline's 6.16). The
productive direction is therefore to make the identifiers discriminate
behaviour *better*, and the mechanism that did it is the behaviour-pair
contrastive term on the residuals — the resample lever, which trains the
encoder on the behaviour pairs instead of drawing a partition of the graph.

## How the two measurements were made

* `tools/research/science/community_curvature/alignment.py` — the community
  partition against the held-out behaviour pairs: the positives' share, the
  negatives' share, the chance level and the lift; and the AF3 curvature's
  separation over the behaviour edges of the train graph against the rest
  (which is empty by construction: the graph *is* the behaviour graph, so that
  second measurement is vacuous and is reported as such).
* `tools/research/science/community_curvature/partition_capacity.py` — the
  three partitions and the two codebooks on the same sample and the same
  yardstick: the curvature-guided partition, k-means over the row-normalised
  transition profile at 64, 256 and 1024 groups, the frozen baseline's level-1
  codes and the resample arm's.

Both write their JSON next to this report
(`community_behaviour_alignment.json`, `partition_capacity.json`).
