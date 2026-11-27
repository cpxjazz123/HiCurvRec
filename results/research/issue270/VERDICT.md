# Issue #270 — verdict

**Question.** Across the archived, independently trained arms, does a geometric quantity that improves
(radial/chart fit to graph distance or to synthetic curvature supervision) stably associate with
recommendation quality (Stage3 test R@10), and can #269's loss be explained by curvature error alone?

**Short answer.** No. The arm-level association is null (Spearman −0.068; family-cluster bootstrap CI
[−0.51, +0.35]; permutation p = 0.66). It does not become positive after conditioning on SID preservation.
The strongest reproducible signal is that geometric "improvement" often arrives **with** SID
compression/regrouping, and the compressed SIDs are what track the recall change. #269's own loss is
consistent with this: its absolute A.1 error improves by 0.249 over #265, but its L2 prefix groups fall
170 → 113 (127 splits, 93 merges) and R@10 falls 0.00736; however, a relative-distortion ruler (J) and
the arm's own fitted κ_t ruler reverse the "improvement", so even the premise is ruler-dependent.
Protocol/family variance dominates arm-level variance (between-family SD 0.00356 vs within-family
0.00094 over 22 families; the only single-card stratum has n = 3).

## Answers

1. **Which curvature improvements are recommendation-irrelevant?** All of them as a group: the rank
   correlation is negative/null, and the item-level pairing is flat (pooled Spearman −0.026; 84 % of
   item-arm cells have zero Δhit, so sign concordance 0.49 is uninformative). Individual arms do show
   the expected direction (e.g. `g4_closure_A` +0.00031 with +1.10 A.1 improvement; `i234_A5`
   −0.0121 despite +4.32 improvement), but the pattern is not stable across families or rulers.

2. **Which structures associate stably?** Protocol and mechanism family, not curvature: 39 four-card
   arms span ±0.017 R@10 while 3 single-card arms are all negative (ρ = −1 with n = 3, a degenerate
   stratum). Family-cluster bootstrap, leave-one-family-out, and the between/within SD comparison all
   say the same thing.

3. **Is the SID an observable intermediate?** Yes, descriptively: curvature improvement ↔ SID
   compression ρ = −0.281; SID compression ↔ ΔR@10 ρ = −0.472; partial curvature → ΔR given SID
   ρ = −0.253. This is a mediation **pattern**, not a causal estimate (no intervention, no shared
   protocol across the pairs). The pre-registered conditional test (P2) is not supported: the
   preserved stratum is ρ = +0.065 (n = 37), the compressed stratum ρ = −0.90 (n = 5), and the
   difference CI [−0.85, +1.44] includes 0.

4. **Is there a conditional relation, and counterexamples?** No stable conditional relation. At the
   class level, #265 shows the counterexample (class mask 5: A.1 improves −5.15 yet hits rise +0.00175),
   while #269 loses hits in every class stratum despite improving A.1; the item-level sign agreement
   is 0.516 with 81 % ties. If the relation exists, it is not identifiable from these independent runs.

5. **What to optimise next?** Not another observational arm sweep. The next experiment must hold SID
   capacity/assignment fixed and vary only the geometric quantity (or vice versa), in one protocol,
   with the same evaluator, so the intermediate is controlled rather than correlated. The concrete
   design is written below; it is a proposal, not an executed experiment.

## Rulers

| arm | A.1 MAE (Σ ruler) | J (Σ ruler) | J (κ_t ruler) |
|---|---:|---:|---:|
| `TIGER_RQ-VAE` | 7.309 | 0.692 | 1.173 |
| `TIGER_euclid_local_curve_supervision_rq` (#265) | 2.246 | 0.183 | 0.168 |
| `TIGER_relative_curvature_ranking_rq` (#267) | 2.204 | 0.175 | 0.169 |
| `TIGER_calibrated_kappa_rq` (#269) | 1.997 | 0.326 | 0.288 |

The absolute MAE and the relative J disagree on #269 vs #265; the κ_t ruler (the chart #269 actually
trained against) also ranks #265 better. Any "geometry improved" claim must name the ruler; none of the
three rulers associates with R@10 across the cohort.

## Pre-registered outcomes

| claim | outcome |
|---|---|
| P1 monotone curvature → R | **not supported / not identifiable** (null; CI includes 0) |
| P2 conditional on SID preservation | **not supported** (difference CI includes 0; sign flips between strata) |
| P3 #269 loss via SID compression | **supported only on the Σ absolute ruler**; reversed by J and κ_t; no causal proof |
| P4 family/protocol dominance | **supported** (between/within SD ≈ 3.8×; single-card stratum degenerate) |

## Coverage and limitations

- 204 Stage3 runs with `test_final.json`; 42 usable geometry arms (exact SID + checkpoint + complete
  57,439-row prediction snapshot); 12 `record_verified` against the immutable run's recorded sha256,
  30 unverified records (kept, flagged).
- All 42 arms were re-evaluated under one target pair set (57,439 rows) and one graph asset; 41
  distortion blocks cover 20,000 sampled rows × 256 references (+ the missing #267 recomputed).
- Scale outliers: several arms' q drift far from the target radius (A.1 MAE in the hundreds); the
  Pearson +0.41 is carried by them and contradicts every rank statistic. Rank statistics are the
  pre-registered primary; absolute per-arm MAE values for those arms are not interpretable as "worse
  geometry".
- The single-card stratum (n = 3) is too small to support a protocol interaction; it is reported as a
  stratum, not as evidence.
- Item-level tests use target-item pairs only (explanatory), not the full test predictions; 84 % of
  item-arm cells have zero Δhit, so sign concordance is dominated by ties and is not used as support.

## Proposed causal test (next round, gated by §19g and the §121 scope gate)

Hold the SID layer fixed and vary only the geometry at **export** time, then check whether the
intervention moves R@10:

1. Freeze one SID-producing Stage2 checkpoint (e.g. #265) and its three codebooks.
2. Produce three SID exports from the **same** encoder output and the **same** codebooks/capacity:
   (a) the arm's native Euclid assignment; (b) the same assignment under the frozen κ_t chart;
   (c) a degree-preserving randomized-geometry control. Only the distance used for assignment changes.
3. Report pairwise SID overlap and collision; keep the arms whose SID differs materially (otherwise the
   geometry has no observable intermediate and the question is moot at this checkpoint).
4. Run Stage3 on the surviving exports under one protocol and compare R@10 paired by user/item
   against the native export; the only changed variable is the assignment geometry.

If the exports collapse to the same SID, the conclusion is that geometry does not reach the SID at this
checkpoint and no recommendation claim is possible; if they differ, the paired Stage3 difference is a
causal estimate for the export geometry. Training-time geometry remains untested by this design and
must be a separate registered arm, not another observational sweep.
