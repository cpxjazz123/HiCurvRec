# Mixed curvature and per-graphlet geometric bias: what the recorded experiments support

Scope: this note reads the persisted Stage3 final records and the Stage2 layer
metrics of the arms that were actually run in this worktree. Nothing was
trained, replayed or selected here. The fixed final test (57,439 rows) has
already been read by earlier rounds, so every number below is an exploratory
comparison with that bias stated, not an unbiased generalisation estimate.

Machine-readable evidence: `geometry_contrasts.json` (contrasts and noise
floor), `../rounds/r02/analysis/SID_GRANULARITY_DIAGNOSIS.json` (the
SID-granularity diagnosis of this round).

## 1. The protocol noise floor

Twelve arms were launched twice against the same Stage2 SID. Their final
Recall spreads are:

| statistic over the 12 repeated arms | value |
|---|---|
| minimum spread | 0.000052 |
| median spread | 0.000949 |
| maximum spread | 0.016139 |

The two largest spreads (`TIGER_RQ-VAE` 0.015338, `curvature_RQ-VAE` 0.016139)
come from cross-generation baseline re-runs, i.e. implementation drift rather
than seed noise. The same-SID repeat of the mixed arm `TIGER_mixed_rq_F0`
(0.057278 vs 0.057226) spreads 0.000052. A difference of one arm therefore
has to be read against roughly 0.001 (median) to 0.016 (worst historical).

## 2. Mixed curvature versus pure Euclidean, at equal capacity

| arm | geometry | final Recall@10 |
|---|---|---|
| `TIGER_mixed_rq_F0` (two runs) | E16 x H16 mixed, alpha = 1/2 | 0.057278, 0.057226 |
| `TIGER_mixed_rq_E` | E16 x E16 pure Euclidean twin | 0.054197 |

Difference: **+0.003055**. That is 3.2x the median repeat spread and 59x the
tight same-SID spread, but smaller than the worst historical cross-generation
drift. Within one protocol generation the comparison is decisive: the mixed
space is the better space at equal capacity.

Corroboration inside the family: `TIGER_struct_mixed_rq` (mixed space plus
structural supervision) measures 0.057905, the highest final Recall recorded in
this worktree.

## 3. Do different graphlets need different geometric biases?

Two independent families vary only the role/geometry correspondence.

| contrast | left | right | difference | vs median noise |
|---|---|---|---|---|
| per-role geometry vs one global bias | `TIGER_role_geom_rq` 0.055380 | `TIGER_role_geom_global_rq` 0.055224 | +0.000157 | below |
| per-role geometry vs shuffled role assignment | `TIGER_role_geom_rq` 0.055380 | `TIGER_role_geom_shuffled_rq` 0.054266 | +0.001114 | above |
| local per-prefix curvature weight vs global weight | `TIGER_mixed_rq_L` 0.056007 | `TIGER_mixed_rq_G` 0.055711 | +0.000296 | below |

The Stage2 structure changes far more than the Recall does. For the same three
role arms, the layer metrics record L1 hard error rates of 3.85 (per-role),
5.26 (global) and 4.51 (shuffled), with 3421 / 4920 / 4386 distinct two-code
prefixes. The role-to-geometry coupling therefore demonstrably reorganises the
partition, and the shuffled control shows that the organisation follows the
graphlet role rather than the mere presence of per-item geometry.

But the Recall translation is at the protocol noise floor: one of the three
contrasts exceeds the median spread and none exceeds the worst one.

## 4. What actually predicts Recall

Pairing 14 arms by their SID file and recomputing the layer metrics gives
correlations with final Recall@10 of +0.838 (absolute L1 prefix carry),
+0.799 (L2), +0.816 (L3), +0.871 (refinement gain in bits), -0.911 (number of
distinct two-code prefixes) and +0.840 (three-code collision). Coarser,
better-populated level-1 partitions carry more of the validation target.

This is the mechanism-level lever that the mixed family already exploits and
that the #255/#256 lineage loses: `TIGER_mixed_rq_F0` carries 15.99% of the
validation targets inside the same L1 prefix with 4151 distinct two-code
prefixes, whereas the two arms of this lineage carry 6.67% and 6.43% with
8559 and 8840 prefixes, i.e. an almost item-unique encoding (three-code
collision 0.065, quantisation error 0.0077).

## 5. Conclusion

* **Mixed curvature is more suitable.** At equal capacity and under one
  protocol, the mixed E16 x H16 space beats the pure E16 x E16 space by
  +0.003055 final Recall@10, reproduced across two Stage3 launches of the same
  mixed arm, and the best arm in the worktree (`TIGER_struct_mixed_rq`,
  0.057905) is also a mixed arm.
* **Different graphlets do need different geometric biases, and the geometry
  does follow the role.** The per-role bias produces a measurably different
  partition than a global bias (L1 hard ER 3.85 vs 5.26) and than a permuted
  role assignment (3.85 vs 4.51), and both Recall contrasts are positive
  (+0.001114 against the shuffled control, +0.000296 for the local weight).
  The direction is consistent across two independent families.
* **The magnitude of the Recall effect is not yet established.** All three
  per-role contrasts sit at or below the protocol noise floor (median 0.000949),
  and no arm of any family reaches the 0.065 threshold. The honest statement is
  "the role-to-geometry coupling is real and directionally positive, and the
  Recall gain it produces is smaller than the protocol noise", not "per-graphlet
  bias has been shown to raise Recall by X".
* **The decisive remaining lever is level-1 granularity**, not the curvature
  schedule: this round's cadence hypothesis (refresh the maximal global module
  budget every 50 native epochs) moved the final Recall by -0.000070 and the
  validation Recall@10 by -0.000331 relative to the single refresh, so the
  schedule is falsified, while the granularity correlate spans 0.043 to 0.058
  of final Recall across arms.

## 6. What would settle the remaining gap

One mechanism whose role-to-geometry coupling acts on the level-1 commitment
itself (role-aligned level-1 codes under the unchanged Sinkhorn balanced
assignment and the frozen 3000/150 recipe), measured on the validation split,
with a preregistered falsifiable prediction. The expected signature is an
absolute L1 prefix carry in the 0.10-0.16 band (against 0.064 for this
lineage), a two-code prefix count near 4000 (against 8840) and a final Recall
difference above the 0.001 median spread.
