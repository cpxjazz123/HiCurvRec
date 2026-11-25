### R05 admission, four-card performance and module gradients — all measured on the final source

**Machine admission** (`checks.run_all`, the frozen §19f gates): **PASS**
- `entry`: real one-epoch trainer run and exact SID export — `levels 256/256/256`, collision `0.054053`, inside the Stage3 gate.
- `behavior_graph`: raw train, split, graph, audit source and training dependencies verified.
- Evidence `results/research/issue258/ADMISSION.json`, sha256 `a1b734e3fe47e85ffb0cc3a7c052d9b8d18a1af1c82d5e5f2c0b1beec3375b67`; command `checks.run_all(candidate tree)`, exit 0, 78.29 s.

**Four-card performance** (`benchmark_entry.py`, actual world4/global4096, 60 complete inclusive epochs, window 31–60):
mean **0.693481 s/epoch**, median `0.6597`, p95 `0.9247`; **ceiling 0.8 PASS**;
0.4 target and 0.5 expected **not met**, recorded truthfully; predicted 3000-epoch time `2067 s`.
The report's source binding matches the current tree on **136 files, 0 mismatches**.
Evidence `results/research/issue258/PERFORMANCE.json`, sha256 `8a7e2778c73558fbafa2b4310ef59fab3744d99fcd3a7b7c736ce4147bfffc8b`.

**Module gradients and updates** (real four-rank first batch of 1024/rank, real reconstruction + sampled
structural loss; no synthetic loss):
- **18 modules**, every one has a non-zero total-loss gradient and a non-zero optimizer update;
- the level-2 **role table** receives both VJPs: original `2.24e-7`, structural `0.03847`, update `8.01e-6`;
- the four-rank real-loss budget regression passes on every rank (maximal λ at the active bottleneck);
- four ranks agree on the completed parameter hashes.
Evidence `results/research/issue258/MODULE_CHECKS.json`, sha256 `661bbe7963cf00d23512967f9fe5fa732618160d8333a839d233c6b3af206164`.

**Dependency-aware validation**: `tools/iteration/depgraph.py` plans six units
(`pipeline_units`, `candidate_suite`, `admission_entry`, `behavior_graph`, `performance`, `module_gradients`);
**6 reuse / 0 rerun**, each with a PASS receipt bound to its fingerprint
(`results/research/issue258/VALIDATION_PLAN.json` sha256 `b2bcace56415f41e18cb296414c2807d982bd90e603ac763a21ff1e08a8afc11`;
receipts under `results/research/issue258/receipts/`).

**Final source**: 79 `.py`/`.json` files, 14 changed + 2 added vs the R04 parent; the earlier full suite on
this source is **163 passed**, exit 0 (`checks/candidate_suite.log` sha256
`8d65bfe1b1a461d9d71d3238aa4c3d7482f2d9d4907ac566f44f296b5fe64ab5`).

**Infra**: still zero drift on the 220-file public baseline. The only gap is FAULT-1 (validation Recall@10 has no
identity-parameterised entry).

Next and final compute step: reserve `issue256_r05_formal` through the frozen workspace API and launch the single
four-card 3000-epoch Stage2 run; then the frozen 150-epoch Stage3.
