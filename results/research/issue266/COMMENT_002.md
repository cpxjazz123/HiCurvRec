## Issue #266 endpoint + ranking result

Frozen measurement completed. R2 108/109 identity: 196/198 checks, **0 failures**; fixed batch SHA `251dced0…457` and structural-sample hashes identical across arms; token cache and directed-graph/curvature assets match their manifests.

| arm | C (raw unweighted) | G1 (raw L1 codebook) | S |
|---|---:|---:|---:|
| 108 | 0.7775872945785522 | 12.757243390234594 | 0.18293017861055832 |
| 109 | 1.5810964107513428 | 7.112086922093524 | 0.3724475438041919 |

- Original 8: Pearson 0.8569503102754842, Spearman 1.0, Kendall tau-b 1.0, strict adjacency PASS.
- Extended 10: Pearson 0.69046552155332, Spearman 0.793939393939394, Kendall tau-b 0.7333333333333333, **strict adjacency FAIL**, **6 inversions**; violating arms `[100,104,105,106,107,108,109]`.
- Required interval `0.48640629657478884 < S_108 < S_109 < 0.5096899206435366` fails: both new scores are below the lower bound. `S_108 < S_109` is true; recall order is correct.
- Adjacent violation: `105 -> 108` (`S` drops from 0.4864063 to 0.1829302).
- 109 uses a different structural objective (Euclidean selection + local mixed-curvature final-q target), so its native C/G1 is an external-transferability check, not a same-objective scale claim.
- Tests: `pytest -q results/research/issue266/tests` → **15 passed**.
- Evidence: `results/research/issue266/{ENDPOINT_108.json,ENDPOINT_109.json,SCORES.json,SCORES.csv,REPORT.md,CHECK_1.md,VERDICT.md}`.
