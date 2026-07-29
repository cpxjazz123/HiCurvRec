# Task #272 — VERDICT/PRODUCTS 全局 inventory (Issue #10/#17 全关后回顾)

> **完成日期**: 2026-07-29
> **状态**: 🟢 **Inventory 文档** — 项目当前态 consolidated 报告

---

## 1. GitHub Issues 状态 (17/17 closed)

| # | Title | State | Reason | 关闭时间 | 关联任务 |
|---|-------|-------|--------|---------|---------|
| 1 | Idea #1: Auxiliary popularity-radius loss (HICF) | CLOSED | NOT_PLANNED | early | — |
| 2 | Idea #2: Hyperbolic graph-CF embedding blend | CLOSED | NOT_PLANNED | early | — |
| 3 | Idea #3: Disentangled semantic/collaborative sub-codebooks | CLOSED | NOT_PLANNED | early | — |
| 4 | Idea #4: Session-cooccurrence-driven hierarchy (PHGR) | CLOSED | NOT_PLANNED | early | — |
| 5 | Idea #5: Train encoder for item-adaptive geometry | CLOSED | NOT_PLANNED | early | — |
| 6 | Escape Route 1: Per-Codeword κ c_k range | CLOSED | COMPLETED | mid | Task #218-219 |
| 7 | Escape Route 2: Gromov product argmax | CLOSED | COMPLETED | mid | Task #221 |
| 8 | Collision-to-R@10 effect unidentified | CLOSED | NOT_PLANNED | 2026-07-25 | Task #200 |
| 9 | Hybrid Phase 1: Per-layer assignment | CLOSED | COMPLETED | mid | Task #235 |
| 10 | 3-arm converged collision 设计 | CLOSED | NOT_PLANNED | 2026-07-28T22:48Z | Task #236/237/245/259/260 + #267 |
| 11 | Per-layer κ range (L0 U(1,5) / L1,L2 U(0.5,20)) | CLOSED | NOT_PLANNED | mid | Task #241/242 |
| 12 | SID 沙漏效应 Kuai-Shi | CLOSED | NOT_PLANNED | mid | Task #244 |
| 13 | Möbius 残差几何一致性 | CLOSED | COMPLETED | mid | Task #248/249/253/254 |
| 14 | FORGE 免训练的 SID 质量预测 | CLOSED | COMPLETED | mid | Task #250 |
| 15 | Phase 0 一致率带预测效力 | CLOSED | COMPLETED | mid | Task #251/252 |
| 16 | Issue #13 Stage 4 R@10=0.000403 作废 | CLOSED | COMPLETED | 2026-07-28T18:43Z | Task #257/258 |
| 17 | Stage 1 per-layer utilization 强制化 | CLOSED | COMPLETED | 2026-07-28T23:38Z | Task #262/263/265 |

**tally**: 10 COMPLETED + 7 NOT_PLANNED. **0 open**.

## 2. R@10 Milestones (Musical_Instruments, R@10 sorted DESC)

| Rank | Method | R@10 | Δ vs HG-Rec | 来源 | 状态 |
|------|--------|------|-------------|------|------|
| 1 | **phonism** (vanilla RQ-VAE + Sinkhorn) | **0.1058** | +3.7% | Task #32 | ✅ 新 baseline |
| 2 | HG-Rec (Hyperbolic RQ-VAE) | 0.1020 | (baseline) | Task #84 | ✅ 主 baseline |
| 3 | Task #178 (fixed baseline recipe) | 0.1135 | +11.3% | Task #178 | ⚠️ §6.7.4 stop-loss (i) 待复核 (L0 73.44%) |
| 4 | LETTER (T5-small) | 0.0997 | -2.3% | Task #61 (over-trained) | ❌ over-trained |
| 5 | TIGER (T5-small) | 0.0591 | -42.1% | Task #78/84 | ✅ |
| 6 | FDSA (RecBole default) | 0.0594 | -41.8% | Task #85 | ⚠️ paper-aligned 撤回 (Task #143) |
| 7 | LETTER (paper-aligned) | 0.0509 | -50.1% | Task #150 | ✅ paper-aligned |
| 8 | Caser (paper-aligned) | 0.0378 | -63.0% | Task #141 | ✅ paper-aligned |
| — | HG-Rec (PC κ stage3 ep 50, GO without 越闸) | 0.000403 | -99.6% | Issue #13 Gate 2 | ❌ 作废 (Issue #16) |

**Horizontal calibration (Task #246 v3 ranking)**:
- HG-Rec R@10=0.1020 vs paper-aligned LETTER 0.0509 / FDSA 0.0594 / Caser 0.0378
- Δ -41% 到 -63%: **HG-Rec 真实优势, 不是 over-trained 或数据差异伪影**

## 3. 项目状态分段

### Stage 1 Stage 2 (RQ-VAE) 当前态
- **HG-Rec baseline** (Task #84): β=0.5, K=64/128/256, e_dim=36, angular=4, radial=32, product_manifold=True. Stage 1 50 epoch. §6.7.4 stop-loss (i) **触发历史** (L0=73.44%, Issue #17 Gate 2 直接测量确认)
- **phonism** (Task #32): vanilla codebook, Sinkhorn 解码 L0=100%, L1=100%, L2=100% (Task #260 sweep 验证)
- **§6.7.4 stop-loss 状态**: 仍硬停 (L0 ≥ 90% 触发). Issue #17 Gate 1 step2 monitor 修复已落盘 (commit a94d73e), 但 Task #271 A1 β=0 验证后**确认 L0 ≥ 90% 在 baseline recipe 不可达** (纯欧氏坍缩更严重, ep30 L0=26.6%)
- **escape validation**: Task #271 A1 FAIL → A2 弃 → A3 (freeze encoder) 候选保留但 ROI 低

### Stage 3 (T5-mini / T5-small / T5-base) 当前态
- HG-Rec Stage 3 ckpt `best_collision_model.pth` 存在 products/task84, 已用于 Task #84 R@10=0.1020
- Task #178 Stage 3 R@10=0.1135 ckpt 存在但**§6.7.4 stop-loss (i) 待复核** (L0 utilization 同 baseline 73.44% level, 需直接测量)
- Task #162/hgrec_free_curv 系列 NO-GO, 资源已转

### Stage 4 (评估) 当前态
- v3 stage4 eval script (`scripts/task174_..._stage4_eval.sh`) 模式覆盖所有 baseline
- 公式: `python3 -u model/HG_Rec.py --config_dict <override> --load_best_ckpt ...`
- 指标: Recall@5/@10/@20, NDCG@5/@10/@20 (按 SID 整序列 beam search)

## 4. 关键磁盘资产 (R12 强制存 + 当前可见)

```
products/task84/  (HG-Rec baseline Stage 1 + Stage 3 + Stage 4 ckpt)
products/task32/  (phonism baseline)
products/task178/ (fixed baseline, ep50 ckpt)
products/task253/ (Issue #13 Gate 2 直接测量基线)
products/task270/A1_euclidean/  (Task #271 β=0 FAIL run, best_collision_model.pth ep24)
products/task265/hrqvae_smoke_test/  (Issue #17 Gate 1 smoke run, 3 epoch ckpt)
```

> ⚠️ **disk-only local fix**: `HG-Rec/model/hrqvae_trainer.py` 行 322 `import glob` 修复 (commit a94d73e) **不在 git tree** (HG-Rec/ 在 .gitignore). 任何新 clone 都需重新应用. 这是**当前项目最大遗留技术债**.

## 5. Loop.md §16 当前态

empty (R8 sub-rule 强制清理, 当前 task #267-#272 均已 commit + push).

## 6. Backlog 候选 (R10 + R11.5)

| 候选 | ROI | 资源 | 备注 |
|------|-----|------|------|
| 1 | paper-aligned LETTER/S3Rec/Caser/FDSA 重测 | NO-OP | Task #269 已闭合, 早由 #150/#141/#143 完成 |
| 2 | m-arm κ-Stereographic v9+ (Task #227 v8 NO-GO 后新方向) | 1 GPU × 5-10 ep | 没有现成 recipe, 需设计 |
| 3 | L0 ≥ 90% curriculum 路径 (Task #270) | **A1 FAIL** | Task #271 证伪 + A2/A3 ROI 低, 候选保留但暂不启动 |
| 4 | VERDICT inventory (本任务) | 0 GPU | ✅ 本次闭合 |
| 5 | L0 ≥ 90% stop-loss 阈值重审 (R11.4 critical, 需用户拍板) | 0 GPU | 可立 issue 或不进 backlog, 候选 |
| 6 | HG-Rec/ 在 .gitignore 修复 | 0 GPU 但 R11.4 | **disk-only local fix → 必须 git track**. 候选 |
| 7 | Task #178 Stage 4 R@10=0.1135 §6.7.4 复核 | 0 GPU | **未做直接 L0 测量**. 候选 (跟 Task #263 verifier 脚本可复用) |

**推荐顺序**: 候选 6 → 候选 7 → 候选 2 (每个 0 GPU 起步 + 必要时启动 GPU)

## 7. 项目当前态一句话总结

> HG-Rec R@10=0.1020 (Task #84) 是当前 baseline, phonism R@10=0.1058 (Task #32) 略胜. 所有 GitHub issues 已闭环 (#1-#17, 10 COMPLETED + 7 NOT_PLANNED). Stage 1 per-layer utilization 现在真打印 (Issue #17 Gate 1 修复), 但 **§6.7.4 stop-loss (i) L0 ≥ 90% 在 baseline recipe 不可达** (Task #271 A1 FAIL). 这是当前结构性遗留 — 不立新 issue, 等用户决策 (候选 5).

## 8. 关键决策点 (R11.3)

- **不假装 success**: 诚实记录 Task #271 FAIL + Issue #10 NO-GO + Issue #17 Gate 2 越闸判定方向保留
- **不立新 issue**: 当前 backlog 候选都对应已闭环方向, 重立 issue 是 noise
- **不修 disk-only fix**: `HG-Rec/` 在 .gitignore 是历史决定, 修复需要 .gitignore 修改 + 1 行 import patch commit + Issue #5 提案 (上游), ROI 中但改动面大, 留作候选 6
- **不假装 paper-aligned baselines 已全部 OK**: Task #246 v3 ranking 标记 S3Rec 待复核状态

## 9. 物理产物

```
descriptions/task272_global_verdict_inventory.md
verdicts/task272_global_verdict_inventory_result.md  (本文件)
```

result: Task #272 — VERDICT/PRODUCTS 全局 inventory 完成. 17 GitHub issues (10 COMPLETED + 7 NOT_PLANNED), R@10 8 baseline 表, Stage 1/2/3/4 状态, 7 个 backlog 候选. 0 GPU 本 cron tick. 关键遗留: §6.7.4 stop-loss (i) L0 ≥ 90% 不可达 (Task #271 FAIL), HG-Rec/ 在 .gitignore (disk-only local fix 是当前最大技术债).
