## All six full-reference cells completed; independent reducer format fault repaired

`evaluate_arm` exited0 after201.43s. Both arms ×L1/L2/L3 evaluated every28,391,948lawful triples and24,326centres with full original references, common fixed-three metric and scale0.08433422446250916. `evaluation/FINAL_EVALUATION.json` SHA-256:`6aa757d6453c92860b7e9a8072b0e8f029532c82904c190a26b5e6bfd786d1a7`; complete GPU extraction receipt:`29eb0830d15d052a101fb005bd62a4be17f63b1bcd40bc5f02c3f2cea97f31fe`.

The corrected arm has306two-code prefixes and457three-code tuples, versus #253's10,386and22,826. Raw visible-edge L1 sharing rises0.036180935→0.925719559, but the strict held-out test degree-control rate is0.935374150, above direct-edge0.918651521. Thus raw sharing is not evidence of selective structure retention. Full curvature macro MAE worsens at all levels; micro MAE improves—both are retained, not collapsed into a favorable headline. No Recall conclusion is possible after the gate refusal.

Independent `reconcile_results` first exited1:
```text
ValueError: ...assets/train/visible_forward.npz does not contain a sparse array or matrix.
```
The immutable asset is rawCSR (`indptr/indices/data/shape`), not SciPy `save_npz` format. No asset was rewritten. The reducer now reads its rawCSR arrays directly; hand-answer regressions cover unweighted edges, excluded out-isolates and empty populations. The tests were written before the helper implementation; the original CLI format error is the actual fault reproduction, not a claimed mutation catch. Complete tooling tests now pass64cases (11.12s); the real independent reduction exits0 and reproduces all SID/edge/six-cell headlines within the declared tolerance.

The separate post-export suite passed104candidate +62tools tests before this reducer-only repair. Original pre-dispatch receipts remain untouched. New final-source evidence: `POST_RECONCILIATION_TEST_RESULTS.json` SHA-256:`7d0e1a7f66db347d177a33b5c1ba27cadce22e46713f8c5621508e0a59199e7f`; `RECONCILIATION.json`:`d9757df2ce1d49dbaa6f8a79459a0bc6043268422eafeaedc413ee956bc6afe3`.
