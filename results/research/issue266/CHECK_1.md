# Issue #266 check 1

- Contract frozen before computation: `SCORE_CONTRACT.md`; issue body sha256
  `09c266af34eef4d0ffdb1a70d1dcdc6254b9bcb6acb3b15ba4f400ee4d0c5952`.
- INFRA_MODE: `FROZEN`; no public infra change.
- Training: not required (analysis-only issue; explicit no-training boundary).

## C1–C9

| C | Result | Evidence |
|---|---|---|
| C1 R2 108/109 four pieces and identity | **PASS** | `IDENTITY.json`: 196/198 file-hash checks, 0 failures; 112/113 ledger files; checkpoint, SID, source, token cache, graph, curvature and recall identities match. |
| C2 #262 measurement definition | **PASS** | Both endpoint JSONs: fixed batch sha `251dced0…457`, sample hashes identical (centre 15918; 256 ids; 28 triples; 32 references), same source forward/autograd path; differences documented in `SCORE_CONTRACT.md`. |
| C3 real endpoint C/G1 | **PASS** | `ENDPOINT_108.json` C=0.7775872945785522, G1=12.757243390234594; `ENDPOINT_109.json` C=1.5810964107513428, G1=7.112086922093524; parameter checksums unchanged; optimizer steps 0. |
| C4 native-loss comparability caveat | **PASS** | `REPORT.md` and `VERDICT.md`: 109 has a different structural objective; native values are external-transferability only; no same-objective claim. |
| C5 ten-arm table/components | **PASS** | `SCORES.json`/`SCORES.csv`: ten rows, all C/G1/components/S/Recall; eight historical values reproduced from #262. |
| C6 statistics and edge cases | **PASS** | `SCORES.json`: 8-arm Pearson 0.8569503, Spearman 1, Kendall 1; 10-arm Pearson 0.6904655, Spearman 0.7939394, Kendall 0.7333333; strict adjacency FAIL; 6 inversions; no tied recall or tied-S pairs; reference tests 15 passed. |
| C7 persistence and reference | **PASS** | execution records under `logs/execution/`; `reference.py`, `tests/`; logs `verify_bundle.log`, `aggregate.log`, `reference_tests.log`. |
| C8 no training/parameter/SID/infra change | **PASS** | endpoint checks: parameters_unchanged=true; no optimizer; SID/checkpoint hashes match; end snapshot diff to be recorded below. |
| C9 conclusion and delivery | **PASS** | `VERDICT.md` separates prediction continuation, scale comparability, gradient sensitivity and optimization benefit; commit/push/close after this check. |

## Key result

The original eight-arm monotonic relation reproduces exactly, but the extension
fails. The new score values are below the issue's historical interval:

```
S_108 = 0.18293017861055832
S_109 = 0.3724475438041919
required: 0.48640629657478884 < S_108 < S_109 < 0.5096899206435366
```

The recall order is correct (`Recall_108 < Recall_109`), but the frozen score
chain is false. One adjacent violation (`105→108`) and six inversions are retained
in `SCORES.json`; no arm was dropped or clipped.

## Commands / logs

- `issue266-measure-108`: `logs/execution/checks/issue266-measure-108/20261126T052638497524Z_660066_21981500/run.log`
- `issue266-measure-109`: `logs/execution/checks/issue266-measure-109/20261126T053005888431Z_692079_4e7b975d/run.log`
- `verify_bundle.py`: `logs/verify_bundle.log`
- `aggregate.py`: `logs/aggregate.log`
- `pytest -q tests`: `logs/reference_tests.log` (15 passed)
