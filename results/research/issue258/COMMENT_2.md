### R05 candidate implemented and verified locally — Infra untouched

**Mechanism** (`stage2_RQ-VAE/TIGER_role_conditioned_l2_rq`, copied from the frozen R04 tree):
the per-role mixture `m_r = softmax(theta_r)` now enters the **level-2** cost
(`D_2(i,c) = Σ_e m_{r(i),e} d_e(residual_i^2, c)`); levels 1 and 3 use the item gate. The gate keeps two learned
rows, row 0 for level 1 and row 1 for level 3, so no query row is left with an identically zero gradient
(`CONTRACT.md` Amendment 1). `role_table.json` is byte-identical to R04 (`32f986da…`).

**Delta vs the R04 source**: 79 `.py`/`.json` files, **12 changed + 2 added, 0 removed**
(`results/research/issue258/SOURCE_DELTA.json`). The 12 are the arm name, the two-row gate, the level-2 routing,
the round rebinding of the benchmark/budget entries and the five affected mechanism tests; the 2 additions are the
independent routing reference and its test.

**Tests** (TDD: first 10 failures on the R04 routing, then green):
- full candidate suite: **163 passed**, exit 0, 345.84 s —
  `results/research/issue258/checks/candidate_suite.log` sha256 `fbcc27f8ebc1d9c4bf081c050b8149e05cb2e08971abe4a3c1425216a9b46040`;
- public pipeline unit suite: **14 passed**, exit 0 —
  `results/research/issue258/checks/pipeline_units.log` sha256 `7ca1357b1585c9070d3286487daabd5a76755cd024e392d053dd3159aa102b37`.

**Mutations** (`MUTATIONS.md`): **13/13 caught by their designated assertions** — numeric 4, structural 5,
routing 4. The new routing mutation puts the role mixture back on level 3 (R04); the pre-mutation checker passes
and the mutant fails on the level-2 role assertion.

**Dependency-aware plan**: six units (`pipeline_units`, `candidate_suite`, `admission_entry`, `behavior_graph`,
`performance`, `module_gradients`) planned through `tools/iteration/depgraph.py`; every unit reruns because this
is a new source generation — no stale receipt is reused
(`results/research/issue258/VALIDATION_PLAN_PRE.json`, receipts under `receipts/`).

**Infra**: the 220-file public baseline recomputed after the edits — `changed []`, `missing []`. No tool, asset,
generator, evaluator, launcher or Skill was modified by this round.

Next: `checks.run_all` admission, four-card benchmark under the 0.8 s/epoch ceiling, four-card module-gradient
regression, then the single `issue256_r05_formal` Stage2 run. The only known Infra gap remains FAULT-1
(validation Recall@10 has no identity-parameterised entry) in `results/research/issue258/INFRA_FAULTS.md`.
