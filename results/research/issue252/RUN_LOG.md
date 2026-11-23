# Issue252 execution log

- Title: Complete attention-gated sectional RQ-VAE end-to-end: resolve graph admission, restore ~0.4s/epoch, finish Stage2 and Stage3.
- Issue body SHA256: `3e9b8ca4018cd9d0a43648efc42cd6a12ef27bc892fdb981590030bea37849eb`.
- Initial source HEAD: `98e6f72e772bf2807dcc794649faf6edf4cc2895`.
- Fourteen original checkboxes are frozen in ACCEPTANCE.json; all are tracked separately.
- **Formal training required.** Issue goal and closing conditions explicitly require one four-card Stage2 3000 epochs and its corresponding Stage3 150 epochs/final test/R2 bundle. Existing #251 diagnostics cannot satisfy these clauses. Formal dispatch remains prohibited until graph admission, mathematical/gradient/DDP/entry tests and >=30 inclusive epochs mean<=0.5s pass.
- assets/ was inspected before any construction: existing v6 manifest68 assets, original graph/features/full28391948-row supervision are reused by hash. No binary regenerated.
- Owner now authorizes graph reconciliation and prospective declared training approximations; #251's former policy blocker is not assumed to persist unchanged.
- Initial host: four NVIDIA L40S, each46068MiB/free, ports50202/50201 not listening. Original-path four-rank profiling was launched through `launcher.launch` by `python3.9 -m tools.issue252.launch_profile`; it is diagnostic, not formal training or an epoch-speed acceptance run. Its final source-bound rank reports are still pending.
- C1 PASS: independent raw train reconstruction proves Case A;62501 missing supports split31100/31241/160, unexplained0 and integer weight residual0. Receipt `assets/v7_graph_contract/audit.json:0918afe35cddf209441fff4fe80a558a2d562e14027326d124ec2f51000acb1b`. Correct frozen binaries were not regenerated. COMMENT_003 publishes the result.
- C3 regression subchecks: named graph11 tests and cache10 tests PASS; original42-test batch retained its41 PASS/1 incomplete-mock fixture ERROR, then the fixture was corrected without changing production validation. Old-generation and in-flight graph identity regressions were observed red before the source fix. Actual MCP/entry admission is still pending.
- Historical30-epoch windows were recomputed by `tools/issue252/historical_timing.py`; old timers exclude post-loop communication/evaluation. Missing GPU/executed-source identity is explicit in HISTORICAL_TIMING.json; these are not new inclusive speed credentials.
- Selective commit guard: `python3 tools/commit_guard.py --max-paths 850 --max-lines 300000` returned0 for732 paths/245245 lines. Elevated limits accommodate the pre-existing shared worktree; only this issue's owned paths are selected, not unrelated changes.

All progress comments and evidence will be persisted here; no checkpoint file is an exit condition. No formal or diagnostic run is counted until actual process/results and source-bound artifacts exist.
