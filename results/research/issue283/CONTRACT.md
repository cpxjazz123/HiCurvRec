# Issue #283 — CONTRACT (frozen 2026-11-28)

Analysis-only, INFRA_MODE=FROZEN. All formulas, thresholds and candidate lists below were frozen
before any statistic was computed. Issue constants only; no implementation may relax them.

## 1. Data

| item | path | sha256 |
|---|---|---|
| triplet shards (13) | `assets/shards/shard_000..012.npz` | per `assets/manifest.json` (28,391,948 rows, all `valid`) |
| query index | `assets/index.npz` | `a5eb2935b258024500d3558c0256238aa342bd1ade52dd0141208c1730e656e1` |
| visible directed graph | `assets/train/visible_forward.npz` | `ac7155d1424a6a346da9b8ed6e9413b81d43223a377308af0c8ba35b6b773029` |
| visible undirected projection | `assets/train/visible_undirected.npz` | `097a719a9f002924…` (manifest) |
| item ids | `assets/train/item_ids.npy` | manifest |
| 253 SIDs, 129 labels, folds | `results/research/issue276/{SID_TABLE,FOLDS}.json` | `table_sha256=4c9279…` |
| #282 type audit | `results/research/issue282/TYPE_AUDIT.json` | committed in a927098b |

The triplet row is `(centre m, b, c)` with `b < c` from the undirected projection; every row is an
induced 3-node subgraph on `{m,b,c}` in the visible directed graph. Integrity check: recomputing the
6-bit directed code from `visible_forward` and applying `canonical_code` must equal the stored
`structure_class` for **all** 28,391,948 rows (assert; any mismatch fails the run).

## 2. Ordered role triple

Roles are positional: **a = row.b, m = row.centre, b = row.c** (the table's own `b < c` order). The
role order is never re-sorted by degree or id after classification. The six directed bits use the
canonical order `ORDERED_PAIRS = ((0,1),(0,2),(1,0),(1,2),(2,0),(2,1))` over positions
`(m,a,b)`: `m→a`(32), `m→b`(16), `a→m`(8), `a→b`(4), `b→m`(2), `b→a`(1).

Abbreviations: `in1=a→m, in2=b→m, out1=m→a, out2=m→b, e12=a→b, e21=b→a, rec1=in1&out1,
rec2=in2&out2`. The adjacency guarantees `in1|out1` and `in2|out2`.

### 2.1 Role classes (mutually exclusive, exhaustive over the 64 codes)

| class | definition | symmetric outer roles |
|---|---|---|
| `chain` | in1&out2&!in2&!out1&!e12&!e21 | no |
| `revchain` | in2&out1&!in1&!out2&!e12&!e21 | no |
| `ff` | in1&out2&!in2&!out1&e12&!e21 | no |
| `fb` | in1&out2&!in2&!out1&!e12&e21 | no |
| `ff_rev` | in2&out1&!in1&!out2&e21&!e12 | no |
| `fb_rev` | in2&out1&!in1&!out2&!e21&e12 | no |
| `chain_both` | in1&out2&!in2&!out1&e12&e21 | no |
| `revchain_both` | in2&out1&!in1&!out2&e12&e21 | no |
| `fork` | out1&out2&!in1&!in2&!e12&!e21 | yes |
| `fork_e12` / `fork_e21` | same, one outer edge | no |
| `fork_both` | same, both outer edges | yes |
| `merge` | in1&in2&!out1&!out2&!e12&!e21 | yes |
| `merge_e12` / `merge_e21` | same, one outer edge | no |
| `merge_both` | same, both outer edges | yes |
| `recip_a_{in,out}[_e12|_e21|_both]` | rec1&!rec2 (+ in2/out2, outer edges) | no (m↔a is a distinct orbit) |
| `recip_b_{in,out}[_e12|_e21|_both]` | rec2&!rec1 (mirror) | no |
| `recip_both[_e12|_e21|_both]` | rec1&rec2 (+ outer edges) | yes |
| `complete` | all six bits | yes |

Family grouping for the report: chain-family = {chain, chain_both}; revchain-family = {revchain,
revchain_both}; triangle-family = {ff, fb, ff_rev, fb_rev}; wedge-family = {fork*, merge*};
reciprocal-family = {recip_*}; complete. "Orbit role" = `orbit_of_position(actual_code, pos)` from
`tools/graph_infra/graphlet.py` for pos in {m, a, b}: the automorphism orbit of each role in the
canonical pattern. Orbit roles are recorded per row and used as a stratification, never as a filter
that silently drops hard-to-match rows.

## 3. Partitions and LCP

At prefix length k ∈ {1,2,3} (cumulative prefix = first k codewords; the 4th value is the export
collision counter and is ignored):
`ALL = a=m=b`, `AM = a=m≠b`, `MB = m=b≠a`, `AB = a=b≠m`, `NONE = all distinct`. The partition is
derived by transitive equality of prefixes, never by comparing two pairs independently.

Invariants (asserted): (i) the five counts sum to the support exactly; (ii) cross-layer refinement
only — allowed transitions ALL→{ALL,AM,MB,AB,NONE}, AM→{AM,NONE}, MB→{MB,NONE}, AB→{AB,NONE},
NONE→{NONE}; (iii) invariance under an independent random relabelling of the codewords at each
level (statistics unchanged); (iv) for symmetric outer roles AM and MB are reported merged (AMB);
(v) single-level diagnostic = patterns computed on the k-th codeword alone (k=2,3) — reported
separately, never mixed with the cumulative main analysis.

LCP: `lcp_pair = #{k ∈ {1,2,3} : prefix_k(pair left) == prefix_k(pair right)}`. Feasible triples
satisfy `lcp_ab ≥ min(lcp_am,lcp_mb)`, `lcp_am ≥ min(lcp_ab,lcp_mb)`,
`lcp_mb ≥ min(lcp_am,lcp_ab)`; the joint distribution is reported over the feasible set only, with
the infeasible count asserted 0.

### Hand example (equal pairwise rates, different joint)

Four triples, each with a distinct code at level 1 (labels A,B,C):
D1 = {ALL, ALL, NONE, NONE}; D2 = {ALL, AM, MB, AB}.
D1: p_am=p_mb=p_ab=0.5, ALL=0.5, AM=MB=AB=0, NONE=0.5.
D2: p_am=p_mb=p_ab=0.5, ALL=0.25, AM=MB=AB=0.25, NONE=0.
The three pairwise rates are identical; ALL differs by 0.25. Given the marginals,
AM = p_am−ALL, MB = p_mb−ALL, AB = p_ab−ALL and NONE = 1−p_am−p_mb−p_ab+2·ALL, so the joint
distribution has exactly **one** degree of freedom beyond the pairwise marginals — ALL (the triple
intersection). AM/MB/AB are affine functions of ALL and are **not** independent information; only ALL
(and any invertible function of it) may be reported as marginal-independent information.

## 4. Null models (offline, statistics only; no training)

Uniform permutation null (permutes item→codeword assignments, preserving the actual group sizes n_g
of each prefix level): for distinct items, E[share_pair] = Σ_g n_g(n_g−1)/(N(N−1)); E[ALL] =
Σ_g n_g(n_g−1)(n_g−2)/(N(N−1)(N−2)), N=24,587. Degree-stratified null: strata = deciles of the
undirected projection degree (quantile edges from `visible_undirected`); permutation within strata;
E[share | stratum pair] = Σ_g (n_{g,s1}/n_{s1})(n_{g,s2}/n_{s2}); E[ALL | triplet of strata] =
Σ_g Π (n_{g,s}/n_s), aggregated over rows with the precomputed per-(class, stratum-triple) row
counts. Verification: 10 empirical permutations on 3 SIDs × chain/merge confirm the analytic uniform
expectation within Monte-Carlo error (reported). Lift = observed − null (absolute) and observed/null.
Classes with support < 1,000 rows (per SID) are reported descriptively with their support and are
excluded from the predictor; no class is silently dropped.

## 5. #282 negative-sample audit and its replacement

#282 matched-negative coverage: level 2 (type∩degree) 1.35 %, level 1 (degree) 11.2 %, level 0 (any)
87.4 % (`issue282/TYPE_AUDIT.json`). Those item-level negatives were **not** type-matched and are not
reused. #283 uses exact deterministic structural stratification (all rows of each class, exact
counts) plus SID-level matching (§6); no item-level negative sampling, no fallback relabelling.

## 6. SID matching (frozen tolerances)

Candidate pair (i,j) requires all of: |Δ l1_pos_hit| ≤ 0.01; |Δ l1_hard_er| / max ≤ 0.20;
|Δ l2_pos_hit| / max ≤ 0.20; |Δ l3_collision| ≤ 0.01; |Δ l3_refine_bits| / max ≤ 0.10; both SIDs
have class support ≥ 1,000 rows. Pairs are used only when ≥ 3 pairs exist for the contrast; every
pair is reported. Dependence: paired SID clusters (127 units, near-duplicate label clusters merged)
are resampled, never the pairs. Contrast tests: ALL (observed and lift) vs recall, and role
contrasts (chain AM−MB; merge/fork AMB−AB) vs recall; BH FDR over the reported contrasts.

## 7. Predictive protocol (C4)

Baseline A = #277 corrected Ridge6 (features `l1_pos_hit, l1_neg_hit, l1_hard_er, l2_distinct,
l3_collision, l3_refine_bits`; inner-alpha + outer-scaler procedure; expected 0.002972 / 0.667 on the
129 rows). Frozen candidate features (all per-SID, support ≥ 1,000):
`all1_chain` = chain ALL at k=1; `all1_chain_lift` = all1_chain − uniform null; `all3_chain_lift` =
cumulative ALL at k=3 − null; `am_mb_chain` = chain AM1 − MB1; `all1_merge_lift` = merge ALL1 − null.
Selection: for each outer fold, using only its training rows, evaluate each candidate with the inner
folds (the remaining outer-fold ids) by the same procedure; select the single best candidate **only
if** it improves the inner MAE over A; otherwise no feature. Nothing is selected on outer labels.
Primary criterion: paired SID-cluster bootstrap (2,000 reps) 95 % CI of MAE(A) − MAE(new) has
lower bound > 0, and Spearman/Kendall, ranking reversals and >0.06 fp/fn do not worsen. Secondary
report: strict 120/9 split, 13 high/6 strong, low-end overestimation / high-end underestimation,
0.065 NOT_EVALUABLE (no label above 0.065). If the criterion fails, #277 is kept unchanged and the
failure is classified as redundancy / support / no increment. No Stage3 is started either way.
