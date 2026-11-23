# Issue 251 execution log

- Issue: 251 — GNN + curvature-expert cross-attention candidate, superseding issue250's single-vector gate.
- Body SHA-256: `eb05f12aee33f42cfff92568e925e00d4810fce6d78db807a31082611528af2d`.
- Acceptance: C1–C9 copied verbatim in `ACCEPTANCE.json`; live body reread during check round1.
- Contract frozen before mechanism integration: `CONTRACT.md:4c4958596a2b40a2bf177e29780d5eeda199e5d2e3495aeafafd4c265b204892`; paper/official comparison in `PAPER_COMPARISON.md`.
- **No formal training.** Issue goal says “仅实现、验证与交付可训练候选，不启动正式 Stage2/Stage3，不重训 TIGER”; C7 permits short validation only. Four-rank microstep and one-epoch admission smoke are verification, not3000-epoch dispatch. No Stage3 or R2 training-result archival applies.

## Candidate boundary

`stage2_RQ-VAE/TIGER_gnn_attention_sectional_rq` was freshly copied from frozen `baseline/TIGER_RQ-VAE`. The original model class is retained; the public model export now points to a subclass. Semantic 768/latent32, three256-entry books, config and original losses are preserved. Standalone geometry, same-cost surrogate, full-reference curvature and provenance helpers were reused as components, not as a replacement baseline. No issue250 training was repeated. Existing issue250/user trees remain untouched.

## Verification sequence

1. Audited existing67 asset entries by hash, actual frozen assignment code and issue249 interfaces. Reused all binary assets. Added only the declarative v6 token schema under `assets/`.
2. Independent hand-computed token/attention references and geometry identities precede integration. Red logs retain missing-production, NaN-padding, invalid-target-ID and manifest-mutation failures.
3. `UNIT_FINAL.log`:60 tests PASS; `REGRESSION_249.log`:79 tests and24 module checks PASS. Nine deliberate mutations caught by specified numerical/structural/routing assertions.
4. Canonical reference extraction retained the same assertions: `REFERENCE_FINAL.log`,13 tests PASS. No test was weakened to accept a production error.
5. Four-rank controlled proof launched only through `tools.issue251.launch_validation` → `launcher.launch`, seed42/world4. Actual outcome remains pending until rank reports and exit code are recorded.

## Commands

```sh
PYTHONPATH=stage2_RQ-VAE/TIGER_gnn_attention_sectional_rq/tests /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 -m unittest test_reference test_known_answers -v
/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 -m tools.issue251.launch_validation
```

No formal `stage2_start` or `stage3_start` is authorized by this issue.
