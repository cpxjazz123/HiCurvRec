# Issue #266 report

## Question

Use the final frozen checkpoints from R2 bundles 108 (#263) and 109 (#265) to
measure the existing #262 endpoint quantities, append them to the original
eight-arm table, and test whether the frozen C/G1 score remains strictly
monotone with Stage3 `Recall@10`.

No training, optimizer, SID rewrite or mechanism change was performed.

## Identity and measurement

- R2 108: arm `TIGER_frozen_topology_gnn_rq`, 112/112 ledger files SHA-256 match;
  four pieces present; archived recall `0.05189853583801946`.
- R2 109: arm `TIGER_euclid_local_curve_supervision_rq`, 113/113 ledger files
  SHA-256 match; four pieces present; archived recall `0.052594926791900976`.
- `IDENTITY.json`: 196 checks for 108 / 198 for 109, **0 failures**. The 28
  R2-only workspace/metadata files per bundle are listed there; they are not
  local source replacements and do not affect the code/result identity checks.
- Fixed batch: 4096 items, seed 42, SHA-256
  `251dced08f5c42703e2ce66b3c56924e91d14200a15a3771f45c8d8c66f1c457`
  for both. Structural sample: centre 15918, `ids` 256 items, 28 triples,
  reference sample 32; `ids`, `triples_global`, `references_global`, `target`
  and `row_ids` SHA-256 are identical for 108 and 109 (see endpoint JSONs).
- Token cache `4051b902934e72f0a06cd6bb657d67d7eec9c9a1c2304c10a31c848515492837`
  matches `token_manifest.json`; its directed graph and curvature target hashes
  also match. No GNN rerun or token regeneration was performed.
- Each endpoint used the actual final checkpoint and source tree, a real forward
  pass, and `torch.autograd.grad(curvature, parameters)`. No optimizer was
  constructed. Parameter checksums before/after are identical; optimizer steps =
  0.

### Endpoint values

| arm | checkpoint sha256 | C | G1 | C component | G1 component | S |
|---|---|---:|---:|---:|---:|---:|
| 108 | `46ad65fb…` | 0.7775872945785522 | 12.757243390234594 | 0.11126996867884884 | 0.07166020993170949 | **0.18293017861055832** |
| 109 | `1f8bfecb…` | 1.5810964107513428 | 7.112086922093524 | 0.33732020746703284 | 0.03512733633715907 | **0.3724475438041919** |

`C` is the raw unweighted curvature loss; `G1` is the raw unweighted L2 norm over
all L1 codebook parameters. Neither is clipped, weighted by `curvature_weight`,
nor replaced by a construction/fixed-evaluation loss. Full gradient shapes and
nonzero counts are in `ENDPOINT_108.json` and `ENDPOINT_109.json`.

## Ten-arm frozen score

The eight historical `C/G1` values are reused from
`results/research/issue262/MATRIX.json`; the formula is exactly the frozen
issue formula, without refitting or clipping.

| recall order | arm | Recall@10 | C | G1 | S |
|---:|---|---:|---:|---:|---:|
| 1 | 101 `TIGER_three_expert_attention_rq` | 0.046431866850049616 | 0.38207152485847473 | 9.739098398312 | 0.052128154424974836 |
| 2 | 104 `TIGER_role_conditioned_l1_rq` | 0.0490085133794112 | 0.6462003588676453 | 21.821778420434185 | 0.20462878183263428 |
| 3 | 100 `TIGER_gnn_attention_sectional_rq` | 0.0492870697609638 | 1.2992827892303467 | 1.6841177967230931 | 0.25803792534182934 |
| 4 | 107 `TIGER_topology_residual_l2_rq` | 0.049600445690210486 | 1.0912375450134277 | 13.289426200724163 | 0.27461306469875635 |
| 5 | 106 `TIGER_role_conditioned_l2_rq` | 0.05034906596563311 | 0.8347048759460449 | 30.384460678370896 | 0.31307429510426327 |
| 6 | 105 `TIGER_role_conditioned_l3_rq` | 0.05151552081338463 | 1.8262606859207153 | 14.063595328730889 | 0.48640629657478884 |
| **7** | **108 `TIGER_frozen_topology_gnn_rq`** | **0.05189853583801946** | 0.7775872945785522 | 12.757243390234594 | **0.18293017861055832** |
| **8** | **109 `TIGER_euclid_local_curve_supervision_rq`** | **0.052594926791900976** | 1.5810964107513428 | 7.112086922093524 | **0.3724475438041919** |
| 9 | 103 `TIGER_periodic_budget_prefix_rq` | 0.05266456588728913 | 1.7692334651947021 | 20.140511086361993 | 0.5096899206435366 |
| 10 | 102 `TIGER_module_budget_prefix_rq` | 0.05273420498267727 | 3.225719451904297 | 32.588647951083324 | 1.0 |

The full machine-readable table and components are in `SCORES.json` and
`SCORES.csv`.

## Statistics

| population | n | Pearson | Spearman | Kendall tau-b | strict adjacent | inversions |
|---|---:|---:|---:|---:|---|---:|
| original 8 | 8 | **0.8569503102754842** | **1.0** | **1.0** | PASS | 0 |
| extended 10 | 10 | **0.69046552155332** | **0.793939393939394** | **0.7333333333333333** | **FAIL** | **6** |

The original eight reproduce the historical result: Pearson ≈ 0.857, Spearman=1,
Kendall=1, no tie or inversion. The extension does not preserve it.

The only adjacent strictness violation is:

```
Recall: 105 = 0.05151552081338463  <  108 = 0.05189853583801946
Score : 105 = 0.48640629657478884  >  108 = 0.18293017861055832
```

Six inversions (lower-Recall arm has larger S) are retained in
`SCORES.json`; their arms are `[100, 104, 105, 106, 107, 108, 109]`. There are
no tied Recall pairs and no tied-S pairs in the observed ten-arm table.
The issue interval is:

```
0.48640629657478884 < S_108 < S_109 < 0.5096899206435366
```

Actual: `S_108=0.18293017861055832`, `S_109=0.3724475438041919`. Both are below
the lower bound; `S_108 < S_109` is true, but the frozen chain is false.

## Comparability and optimization interpretation

- 108 and 109 do **not** optimize the same structural objective. 108 is the
  frozen-token #263 objective; 109 changes selection to Euclidean and uses a
  center-local mixed-curvature target on the final hard `q`. Their native C/G1
  values are therefore an external-transferability test of the frozen formula,
  not evidence that the same objective was optimized.
- The native scale difference is visible but not a causal explanation: 109 has a
  larger C (1.5811 vs 0.7776) and a smaller G1 (7.1121 vs 12.7572), yielding a
  higher S than 108 but still below the historical interval.
- G1 is a diagnostic gradient magnitude, not an actual optimization benefit.
  The two new arms do not support a simple "larger gradient norm → larger gain"
  rule (108 has larger G1 but lower Recall@10 than 109), and n=2 is not enough
  to establish a mechanism-level relation.
- The historical scores were already selected post hoc (the issue states this);
  the new arms have known historical recall and are not a blind test. Failure of
  the monotone extension is the valid result; no search, clipping, retry or
  mechanism selection was performed to restore it.

## Reproduction

```
measure_endpoint.py 108
measure_endpoint.py 109
verify_bundle.py
aggregate.py
python3.9 -m pytest -q tests
```

All command logs, exit codes, and command records are under `logs/execution/`;
the independent 15-test reference suite passes.
