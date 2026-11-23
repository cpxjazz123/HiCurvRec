# Issue 249 execution ledger

Frozen issue body SHA-256: f33f9db4a0b341977cdec8ac989c95c44d9443a32f72bbb516831f48c3db4f0b. Source: ISSUE.json.

No training: issue goal says not to start model training, and its prohibition section excludes GNN, Teacher, gating, RQ-VAE, Stage2 and Stage3 training. All runs are read-only inference, numerical diagnostics or asset validation.

Existing assets were audited through the root manifest; no frozen curvature target was overwritten. Added static asset family: assets/v5_gnn_geometry. 67 entries verified by manifest hash. Frozen one-hop egos are loaded from the original manifest-referenced NPZ; two-hop egos remain on-demand.

Commands:
- `bash tools/graph_infra/verify_v5.sh` — regression/numerical/mutation suite and real-data module checks.
- Stage2 interpreter `-m tools.graph_infra.run_fidelity_v5` — deterministic 2000-triple diagnostic, no training.
- Stage2 interpreter `-m unittest tools.graph_infra.tests.test_v5_infrastructure.EgoTests tools.graph_infra.tests.test_v5_heldout` — 19 tests PASS after frozen-ego integration.

Final fixes have red/green tests in test_v5_heldout.py: macro MAE aggregation, chunk-global coordinate/mask recovery, immutable one-hop reuse. Full output: VERIFY_FINAL.log; portable numerical evidence: MODULE_CHECKS.json, MUTATIONS.json, FIDELITY.json. Hash binding: CODE_HASHES.json.

Guard override only for selective drafting in the shared historical worktree: --max-paths 750 --max-lines 300000. Unrelated experiment trees, binary files and SID tables are excluded from this delivery.
