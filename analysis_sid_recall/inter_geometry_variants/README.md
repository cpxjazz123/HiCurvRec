# Inter-prefix geometry variants vs Recall (Issue #135)

Aggregation/conditioning/normalisation variants of the **same** inter-prefix
quotient geometry measures from #133, judged on the same 108 content-hash-matched
Stage3 Recall@10 arms. Offline only: no Stage 2/3 training, no loss, tokenizer,
codebook or interaction-weighting changes, no causal claim.

## Frozen candidate family

`candidates.py` states each candidate's family, formula, exact denominator,
applicable levels and priority **before** the scan; `candidates.py --freeze`
writes `candidate_registry.json` and `candidate_registry.sha256` (pin this hash).
70 entries: 43 explicit + 27 derived (3 statistics x 9 features).

| family | what it adds |
|---|---|
| **A** | per-node local transitivity (mean/median/p10/p90/IQR), `t_v` triangles per node, closed-ego density and ego cycle surplus, cycle rank per node and per edge, per-component cycle surplus, degree mean/p10/IQR/CV and p90/mean, median/mean, Forman quantiles and per-node incident-curvature means |
| **B** | parent-conditioned sibling quotient `Q_l[P]` inside every occupied parent (L2/L3), the same metrics per parent, aggregated unweighted (mean/median/p10/p90), child-count-weighted and item-count-weighted (declared sensitivity); sibling vs cross-parent edge accounting |
| **C** | null-relative `delta`, `z` and `ratio` for nine predeclared features against three designs (global, log2-degree-stratified, parent-conditioned) with 20 deterministic replicates each; `z` only when the null sd is positive |
| **D** | **secondary** multi-item-prefix-only induced subgraph with lost nodes/edges/items reported |

## Graph and statistics

`Q_l` is the complete binary TRAIN-only quotient: every occupied prefix is a node
(singletons kept), an edge iff a TRAIN item-item edge crosses prefixes, no
self-loops, **no cap** (the 320-child test asserts this). The primary cohort is
all 108 matched rows; the 76 `curvature_RQ-VAE` rows are the stability
diagnostic; the other 32 are descriptive only; the strict same-commit subset is
reported when it has 73 rows.

Every candidate x level gets raw Spearman rho, permutation p, bootstrap 95% CI,
Holm over the whole frozen exploratory family, n and coverage. Each candidate is
also paired with its declared upstream baseline on identical rows and resamples
(5000 draws) to give the paired `delta |rho|` CI. The verdict is predeclared:
**GAIN** needs a valid geometry definition, coverage >= 0.95, a paired CI above
zero and a direction stable within the 76 arms; otherwise **NO STABLE GAIN**.

## Files

| file | contents |
|---|---|
| `candidates.py`, `candidate_registry.json|sha256` | the frozen registry and its pin |
| `variants.py` | A/B/C/D computations, exact triangles via `t_v = ((A^2).multiply(A)).sum(1)/2`, sibling graphs, null shuffles |
| `run_measure.py` | per-SID measurement with checkpoint/resume; null replicates; bench |
| `run_stats.py` | all cohorts, Holm, paired |rho| CI, figures, REPORT, verdict |
| `features_by_level.csv|json` | every candidate value per SID and level |
| `group_parent_support.csv` | nodes/edges/parents/sibling edges/induced losses per SID and level |
| `spearman_by_variant.csv`, `statistics.json`, `bench.json`, `completeness.json` | results, runtime/memory, exactness ledger |
| `tests/test_variants.py` | 20 toy-graph tests: path/triangle/two triangles/clique, singleton child, single-child parent, 3+ children with cross-parent edges, 320 children, L3 singleton-heavy subgraph, null design invariants, monotonic rank invariance, join and edge accounting |

## Reproduce

```bash
python3 analysis_sid_recall/inter_geometry_variants/candidates.py
python3 analysis_sid_recall/inter_geometry_variants/run_measure.py --resume
python3 analysis_sid_recall/inter_geometry_variants/run_stats.py
python3 -m unittest discover -s analysis_sid_recall/inter_geometry_variants/tests
```

## Scope notes

Forman-Ricci is computed over every quotient edge with the unweighted
`4 - deg(u) - deg(v)` definition. #134's true-geodesic ORC is **not** imported;
the old clipped ORC correlation is not used as evidence. The 108-arm scan is
exploratory; no formula is revised after inspecting rho, and no held-out family
is claimed as confirmation.
