## Issue #266 completed — negative monotonicity result

- Original 8 arms reproduce the frozen fit: Pearson `0.8569503102754842`, Spearman `1.0`, Kendall tau-b `1.0`, strict adjacent monotonicity PASS.
- Extended 10 arms: Pearson `0.69046552155332`, Spearman `0.793939393939394`, Kendall tau-b `0.7333333333333333`, strict adjacent monotonicity **FAIL**, **6 inversions**; violating arms `[100,104,105,106,107,108,109]`.
- 108: C `0.7775872945785522`, G1 `12.757243390234594`, S `0.18293017861055832`.
- 109: C `1.5810964107513428`, G1 `7.112086922093524`, S `0.3724475438041919`.
- Required chain `0.48640629657478884 < S_108 < S_109 < 0.5096899206435366` fails: both new scores are below the lower bound; `S_108 < S_109` is true. Adjacent violation is `105 -> 108`.
- R2 108/109: 112/113 ledger files and local source/result/token/graph identities verified, 0 failures. Endpoint forward/autograd had `parameters_unchanged=true`, `optimizer_steps=0`; no training, SID rewrite or parameter update.
- 109 uses a different structural objective (Euclidean selection + local mixed-curvature final-q target), so its native C/G1 is an external-transferability test, not a same-objective scale claim. G1 magnitude is diagnostic, not proven optimization benefit.
- Evidence: `results/research/issue266/{SCORE_CONTRACT.md,ENDPOINT_108.json,ENDPOINT_109.json,SCORES.json,SCORES.csv,REPORT.md,VERDICT.md,CHECK_1.md,IDENTITY.json,INFRA_DIFF.json}`; tests `pytest -q results/research/issue266/tests` → 15 passed.
- INFRA_MODE `FROZEN`; `INFRA_DIFF.json` empty (`ok=true`); commit `1b70f7a12d09046cd8581d1718906e269c439a0a` pushed to `main`.
