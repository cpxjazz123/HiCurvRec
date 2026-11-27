# Issue #269 run log

- Issue: #269 — Scale-calibrated triplet curvature: distinguish local geometry from
  global distance rescaling and validate hard SID
- Issue state at freeze: OPEN
- Issue body sha256: `407618e5d1a7336f4cdd0daa09aaa77c880de727f5d6e384f5b298805eb5d6a4`
  (2635 chars) — frozen 2026-11-27.

## INFRA_MODE

`FROZEN`. The issue states: "单 Agent，INFRA_MODE=FROZEN；复用 #268/#265 及已建资产，不开发或修改
公共 infra，不重复构建图数据。" No sentence authorises building or repairing public
infrastructure. Candidate-local offline fitting, caching, diagnostics and tests stay
under `results/research/issue269/` and the candidate tree.

## Does this issue require training

Yes. Steps 3 and 4 require one Stage2 (3000 epochs, seed 42, frozen batch) and one
Stage3 (150 epochs, seed 42, beam 20) whose final `test_R@10` cannot be produced
from existing weights. §4 "一次正式 Stage3 150 epoch … 最终 test 一次，独立复算
Recall/NDCG@5/10".

## Acceptance checklist C1..C15

| ID | Issue requirement | Criterion / output | Status |
|---|---|---|---|
| C1 | 复核 #268 距离定义、资产行 ID、Z_ref、参考点、有效 mask、图最短路径口径、目标函数；区分有向行为边与对称图距离 | `CONTRACT.md` §1-§2; `gate.py` reuses the frozen assets and the edge/2-hop convention | PASS |
| C2 | 冻结 #265 参考 encoder；确定性拟合与诊断拆分；独立参考点迁移；用相同 pair 权重拟合唯一全局 S*；解析验证 1/S；0.068284 可复现，不硬编码 | `analysis/GATE.json`: `scale_reproduced=0.06828365095335823`, relative error `2.86e-7` from the closed-form weighted least squares over the 20,000-row check | PASS |
| C3 | 冻结 S*；明确坐标映射、κ=0 尺度、合法域与双曲边界；不得扩大 κ 范围制造非法坐标；记录不可识别性影响 | `CONTRACT.md` §2-§3; `GATE.json` `grid_min=-0.05`, `grid_max=1.0`, `max_coord_norm=4.3605` | PASS |
| C4 | 相同冻结尺度与样本，比较欧氏、单一全局 κ、逐三元组 κ；以尺度不变形状误差独立诊断；全部离线 | `GATE.json`: `mean_j_zero_b=0.181664`, `mean_j_global_b=0.182597`, `mean_j_triple_b=0.173216`, `mean_j_shape_free_scale_b=0.148579` | PASS |
| C5 | 事前冻结区间/参考集/权重/容差/判据；逐三元组拟合并报告 held-out 改善、分位数、边界/平坦、符号分布、中心内方差、结构分层、参考扰动稳定性 | `gate.py` constants frozen before the run; `GATE.json` + `PER_ROW.npz`; full-data stratification in the cache manifest | PASS |
| C6 | 成功条件：独立 κ 相对全局 κ 的 held-out 平均 J 至少降低 1%，按中心 bootstrap 下界 > 0，参考扰动 Spearman ≥ 0.5；边界比例报告；阈值不事后修改 | `GATE.json`: improvement mean `0.048371`, CI `[0.041905, 0.055416]`, Spearman `0.946566`, boundary A/B `0.1543/0.2773`, `gate.passed=true` | PASS |
| C7 | 全量 shard 处理全部合法三元组，保存 S*、全量 κ、行 ID、失真、诊断/失败标记、hash、时间、缓存校验；抽样只作独立审核 | `kappa_cache_s0p068284/{kappa,j_zero,j_fit}.npy` + `manifest.json` (28,391,948 rows, 115.3 s, 325 MB) | PASS |
| C8 | 未通过则用 κ=0 退化策略；通过则用冻结逐三元组 κ；只训练一种候选 | Gate passed → per-triple frozen κ; candidate tree `TIGER_calibrated_kappa_rq` | PASS |
| C9 | Stage2：保留 #265 点值目标；全部节点距离用固定 κ 与 S*；`L=L_RQVAE+λL_curvature`；λ 用原预算；监督 hard q 之和；尺度/κ/图目标/参考表示冻结 | candidate wiring `w=q/(2S*)`; `STAGE2_REPORT.json` curvature calibration (λ `5.884e-6`, encoder ratio `0.356`); `test_the_coordinate_scale_is_applied_to_the_hard_q` | PASS |
| C10 | κ=0 极限、正负曲率、边界、重复点、参考一致性、索引错位、固定 q 不可调、梯度路由、hard SID 导出一致性；只跑必要测试；持久日志 | 121 candidate tests PASS; 60-epoch batch-4096 probe PASS; MCP smoke entry PASS (`collision 0.04966`), behavior graph `NOT_APPLICABLE` | PASS |
| C11 | Stage2 3000 epoch、seed42、固定 global batch、原协议；记录重建/量化/碰撞(<0.40)/有效秩/码本使用/前缀命中与 ER/Graphlet-Orbit 保真/固定欧氏与三专家误差与排序/自身口径失真/梯度；inclusive 时间；单卡另报 | `STAGE2_REPORT.json` (1213.8 s, 0.392 s/epoch, collision 0.1567); layer record; `OWN_KAPPA.json`; `CURVATURE_FULL_*`; `LOCAL_STRUCTURE.json` | PASS |
| C12 | 一次 Stage3 150 epoch、seed42、beam20、原协议；冻结候选后最终 test 一次，独立复算 Recall/NDCG@5/10；对比 TIGER、父代与 0.065；披露单卡差异与 test 重复访问风险 | `test_final.json` + independent recompute + `analysis/STAGE3_RECALL.json` | PENDING |
| C13 | 分别回答：尺度校准是否解释 #268 退化；独立参考下局部 κ 是否提供额外信息；hard q/SID 是否改善；Recall 是否改善 | `VERDICT.md` §answers | PENDING |
| C14 | 交付 DIAGNOSIS/HYPOTHESES/CONTRACT、资产 manifest、持久日志、代码/专属测试、训练与性能报告、R2 四件套、VERDICT、推送 | all artifacts + R2 + commit sha | PENDING |
| C15 | 负结果如实记录；不把未执行项报 PASS；完成合法 Stage2/Stage3 才宣称端到端；0.065 是目标不是自动达成 | `VERDICT.md`; no unrun item marked PASS | PENDING |

## Mechanism scope

- Candidate tree only: `ARM_NAME`, the calibrated κ cache reader and coordinate scale,
  the κ-stereographic distance and its wiring into the existing point-value curvature
  objective, mechanism tests.
- New research artifacts: `results/research/issue269/` (gate, fit, validation, docs)
  and the new full cache under the candidate tree.

## Reuse plan

- Assets: `assets/train/visible_undirected.npz`,
  `assets/v8_runtime_layout/targets.npy` + `target_offsets.npy`,
  `assets/node_moments.npz`, `assets/item_summary.npz`, `assets/index.npz`,
  `assets/shards/*` (hash-referenced, never rebuilt).
- Z_ref: the frozen `results/research/issue268/artifacts/z_ref.npy`
  (sha256 `a2643a9b…`), the #265 final encoder output.
- Candidate base: the #268 tree (`TIGER_fixed_triplet_kappa_rq`), the direct
  implementation of the same mechanism; the #269 change is the calibrated scale and
  cache, not a new network.
- Baseline results: frozen TIGER / #265 SIDs and recalls read-only.

## Timeline notes

- 2026-11-27: issue frozen; gate constants frozen in `gate.py` before running.
- 2026-11-27: gate passed (4.84 % held-out improvement, CI > 0, Spearman 0.947).
- 2026-11-27: candidate scale/cache change; NaN and floor stall from #268 fixed
  (`project` produced inf in the unselected `where` branch; coincident `sqrt`
  gradient fixed with `_SafeSqrt`). 60-epoch single-card probe at batch 4096 passes.

## Execution record

- Mechanism + Stage2 commit `80466a12` pushed; `origin/main` matches.
- Stage2 run `stage2_bba78cf7c160467da1c8ae7de9b6e26f`; Stage3 run started
  `2026-11-27 13:52:55` (local), single card, 150 epochs, mean 26.3 s/epoch.
- `commit_guard.py` needed `--max-paths 1500 --max-lines 90000` because the worktree already
  carried 1,173 unrelated changed paths (issue264 exports, archived arms, old smoke artifacts);
  the committed selection is the 190 #269 paths only (docs, candidate tree, receipts, the
  #267 tree deletion).
