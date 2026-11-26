# Issue #263 — acceptance freeze and execution log

Issue: `cpxjazz123/HiCurvRec#263`  
Title: Pretrain directed topology GNN with Orbit and sectional-curvature distributions; freeze tokens for RQ-VAE  
Body sha256: computed in `INFRA_SNAPSHOT_START.json` after this freeze.

## Decision

- **INFRA_MODE = FROZEN.** The issue says: “公共 Infra 与历史产物冻结；允许候选专属预训练入口、数据读取、
  token 导出与缓存、测试和报告，不新建公共框架。” There is no authorization to modify public
  infrastructure. Candidate-local code and the token cache are allowed.
- **Training is required.** The acceptance criteria require a new pretrained GNN checkpoint,
  a frozen token cache, a 3000-epoch four-card Stage2 run, a 150-epoch Stage3 run, and
  four-card performance. This is not an analysis-only issue.
- Only one formal pretraining candidate and one Stage2/Stage3 run. No seed search, no
  hyperparameter search, no extra ablation arm. The frozen TIGER baseline is referenced,
  never retrained.

## C1–C12 (verbatim requirements mapped to tests)

| id | requirement | criterion | evidence |
|---|---|---|---|
| C1 | audit existing GNN input/token shape/direction/scale, mask, gate interface; freeze pretraining contract | contract and audit report; test shapes/mask | `AUDIT.json`, `CONTRACT.md` |
| C2 | build frozen role/orbit and curvature distribution targets on TRAIN graph; no test | labels cover all items where evidence exists; empty masks explicit; source hashes | `LABELS.json`, label tests |
| C3 | pretrain the GNN once with only training items/no test; seed42; fixed budget; no parameter search | role/curvature loss and held-out diagnostics; exact final checkpoint | `PRETRAIN.json`, `PRETRAIN.log` |
| C4 | remove head, freeze GNN, full-catalogue token/mask cache; independent direct inference equality | all 24,587 rows, 4×32 tokens, mask; exact order; same checkpoint | `TOKENS.json`, `token_cache.npz` |
| C5 | Stage2 reads tokens by item id; no GNN forward/backward or optimizer entry | call counters = 0; frozen token hash before/after; gate/encoder gradients real | `STAGE2_CHECKS.json` |
| C6 | four-card 60-epoch inclusive benchmark, 31–60 window including epoch 50 | mean/median/P95 and module times; no graph/curvature coverage reduction | `PERFORMANCE.json` |
| C7 | one 3000-epoch Stage2, seed42, global batch4096, final checkpoint; collision < 0.40 | SID/metrics/checkpoint/run identity | `results/stage2_RQ-VAE/<ARM>/`, `METRICS.json` |
| C8 | one frozen Stage3 150-epoch beam20; validation then one final test; recompute | test R@5/10, NDCG; compare frozen TIGER and 0.065 | Stage3 `test_final.json`, `RECOMPUTE.json` |
| C9 | offline pretraining/cache cost and amortisation | separate pretrain/cache/Stage2 epoch times | `COST.json` |
| C10 | R2 four-item archive including pretrain checkpoint/token manifest | four items present with SHA and ledger | `R2_MANIFEST.json`, ledger |
| C11 | public infra zero drift | identical `path:sha256` at start/close | `INFRA_SNAPSHOT_START/END.json` |
| C12 | complete evidence, commit/push/comment/close | issue CLOSED with links | `COMMENT_001.md`, `VERDICT.md` |

## Mechanism modification scope

Candidate tree only:

- candidate-local pretraining script and tests;
- candidate-local `model/structure_gate.py` / `mechanism.py` wiring to a frozen token cache;
- candidate-local `train_rqvae.py` argument-free cache path and source constant;
- no change to public `tools/`, `stage2_RQ-VAE/baseline/`, Stage3 entry, assets, or admission.

**No old top-level mechanism remains**: per CLAUDE.md §1.0a the prior active
`stage2_RQ-VAE/TIGER_topology_residual_l2_rq` is archived before the candidate copy.
`baseline/` and `_archive/` stay in place.

## Current status

- [x] issue read and exact body saved
- [x] INFRA_MODE determined: FROZEN
- [x] C1–C12 frozen
- [ ] assets audit and label build
- [ ] old mechanism archive and candidate copy
- [ ] pretraining
- [ ] token cache/freeze
- [ ] Stage2/Stage3/performance/R2

## 2026-11-26 execution notes (Issue #263)

- Pretraining completed after fixing three implementation defects: calibration was accidentally
  run under `torch.no_grad()` (removed), the degree-baseline array was length-`eval` instead of
  full-catalogue (fixed), and held-out target arrays were passed as NumPy to a Torch CE (fixed).
  `PRETRAIN.json`: lambda=1.278927, cache max reload diff=6.62e-06 (tolerance 1e-4), held-out
  own/dgdv/curve CE = 3.3738/1.7776/1.5138; top1 = 0.1106/0.3433/0.6280.
- Real one-epoch `checks.check_entry` PASS: collision 0.052141 / 0.051938 / 0.054175 across
  repeated checks; 24587 unique identifiers. `check_behavior_graph` is NOT_APPLICABLE after
  removing an unused inherited `BEHAVIOR_GRAPH_NAME` declaration.
- `stage2_start` first refused because the copied parent tree still contained the generated
  `execution_entry.py`; removed it (no references). Second attempt froze the source but the
  installed `assignment_floor.py` was older than the framework floor module; applied
  `patch_tree.apply_floor` (family euclid), verified `launcher.prepare_floor(..., frozen=True)`.
- Formal Stage2 run started: `stage2_ae668bc04dc44d52a8bed5f9408e8e74`, 4 ranks,
  batch 4096, 3000 epochs; result directory under `_runs/.../result`.

## Stage2/Stage3 transition (2026-11-26)

- Stage2 finished at epoch 3000 with `final_collision=0.06076381827795174`,
  `unique_after_extend=24587`, `checkpoint_selection=final_epoch`.
- Export gate result: `VERIFIED`, checkpoint reproduces the SID; 220 items were
  rewritten from the live export so the saved SID is the checkpoint's own table.
  SID sha256 after rewrite: `a4d7241a12d4e8e52710414bb47baf5441cd63456d4774c1e820eb177eba5184`.
- §63 layer record: collision `0.060723146378167336`; L1 positive hit rate
  `0.063894`, L1 negative hit probability `0.007457`, L1 Hard ER `8.5683`;
  conditional G3 remains explicitly MISSING.
- Stage3 started as variant `TIGER_frozen_topology_gnn_rq`; SID quality gate
  showed collision `0.0607 < 0.40`, quant error `0.007375`, effective rank
  `5.011`; quant-error warning is recorded, not waived.
- Performance: 3000 epochs, all-epoch mean `0.319697 s`, first-60 mean
  `0.399032 s`, max rank peak allocated `607,034,880` bytes.
