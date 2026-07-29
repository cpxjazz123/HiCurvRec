# Task #267 — Issue #10 自主决策 A2: 关 issue 走 NoGo

## 用户指示 (2026-07-29)

> "don't ask me any question just do by yourself"

按 R11.3 自主决策原则 + 用户最高优先级指示, Issue #10 走 **A2 路径**: 关 issue, 接受 H1 NO-GO 结论. 不启动 4-arm 单变量设计 (~13h GPU), 不启动 A1 β+Sinkhorn 联合扫描 (~2-4h GPU).

## A2 决策依据 (R11.3 自主选择)

**Issue #10 H1 (Sinkhorn 碰撞率下降 → R@10 上升) 在 vanilla 族内 NO-GO**, 证据链三条:

| 证据 | 任务 | 结论 |
|---|---|---|
| Sinkhorn 旋钮扫描 (Task #260) | Sinkhorn max_iters ∈ {0, 5, 10, 20, 30} 在 #84 baseline vanilla codebook 上 5 iter 即 full convergence; collision_post 全部 0.0; L0/L1/L2 util 100/100/100% | Sinkhorn 旋钮无效 |
| Arm A → Arm B 碰撞率差 89pp, R@10 持平 | Task #237 (Sinkhorn max_iters=10) | 碰撞率砍 89pp, R@10 +0.1% — 不构成杠杆 |
| Task #225 §5 同 collision 跨机制 R@10 差 40% | 9.07% collision 对应 baseline 0.1020 / TIGER 0.0615 / Letter 0.0997 | 同 collision 水平, R@10 横跨 66% |

**最终结论**: `collision_rate` 不是 R@10 的有效杠杆. 跟 §6.7.4 stop-loss (i) (Issue #17 关心的 utilization) 不是同一回事 — Issue #17 主线已 PASS, Issue #10 这条线在拿到证据后**关闭**.

## 反派备选 (R11.3 必须明示)

| 备选 | 资源 | ROI 评估 | 不选的理由 |
|---|---|---|---|
| A1 (β + Sinkhorn 联合) | ~2-4h GPU | 中 | Sinkhorn 已证无效, β+Sinkhorn 联合扫描大概率重演同一个 NO-GO 结论 |
| B (4-arm 单变量 D/E/F) | ~13h GPU | 低 | H1 证据已足, ROI 边际递减 |
| C (用户新方向) | 任意 | 未知 | 用户拒绝新方向 (指示 "do by yourself" 暗示接管) |

**选 A2**: 证据最足 + 资源 0 + 闭环判定明确 + Issue 列表清空, R10 + R11.5 主动推进 backlog 高 ROI 候选的最佳起点.

## 关闭 Issue #10 路径

按 GitHub API 关闭:

```bash
gh issue close 10 --repo WENYULIANG123/GeneRec --reason "not_planned"
```

关闭评论里写明: (a) Gate 0 (口径统一) ✅ PASS (Task #236/#245/#259); (b) Gate 1 (Sinkhorn Arm B) PARTIAL FAIL (Task #237); (c) Gate 1 修订 Sinkhorn 强度扫描 (Task #260) 全 5-30 iters 等价 → Sinkhorn 旋钮无效; (d) 3 条证据汇总: collision 不构成 R@10 杠杆, 接受 H1 NO-GO; (e) 留下的建设性产物: `collision_rate = (N - unique) / N` 单一权威定义 (scripts/task223_stage2_codebook.py:136) + 历史数字重述 + task200 -10.3% retro-label.

## 关键决策点 (R11.3)

- **不再启动 Arm D/E/F 训练**: H1 证据已足 (3 条独立证据), 关闭 issue 而非耗 13h GPU 重证同一结论
- **保留所有 task 产物**: Task #236/237/245/259/260 的 verdict / script / ckpt 都是公开审计资产, 不删
- **Issue #17 已在闭环 PASS**: 跟 Issue #10 关闭无关, 单独路径
- **资源转向 backlog 高 ROI**: 候选有 (1) paper-aligned LETTER/S3Rec/Caser/Fdsa 已 paper-aligned fix 后的代际重跑 R@10 重新基线, (2) m-arm free curv continuation, (3) cross-architecture generalizability 实测, (4) Stage 1 利用率作为新方向 (H2/H3 来自 Issue #17)
- **Issue #9 状态**: 没在 open issues 列表里 (2026-07-29 latest), 已 CLOSED. 不再 trace
- **不假装用户拍 A2**: 用户说 "do by yourself", 我自主决策. 在 verdict 跟 GitHub 评论里**明示这是 AI 自主决策 + 完整证据链 + 反派备选**, 让用户能事后 audit. R11.3 完全合规.

## 物理产物

```
verdicts/task267_issue10_close_nogo_result.md  (本文件)
descriptions/task267_issue10_close_nogo.md
gh issue close 10 --repo WENYULIANG123/GeneRec --reason "not_planned"  (执行)
gh issue comment 10  (关闭前先 post 决策评论)
```

## 后续 (R10 主动推进 backlog)

按 R10 主动推进原则 + user "do by yourself" 指示, 关 Issue #10 后立即推进 backlog 高 ROI 候选:

1. **paper-aligned LETTER/S3Rec/Caser/Fdsa 重跑 R@10** 候选目录 (Task #141/143/148/150/151 已 paper-aligned fix, 但完整 R@10 重测未做; 这是 HG-Rec baseline R@10=0.10204 之外的横向校准)
2. **m-arm free curvature continuation** (Task #227 v8 collision NO-GO 后, 候选 v9+ 新方向)
3. **Stage 1 utilization-targeted 新方向** (Issue #17 新约束下的下游使用, 比如三层 utilization 监控 + 自适应 Sinkhorn)

资源分配: 4 GPU 全空闲, 推进 (1) 是最稳路径 (paper-aligned 已闭环, 只需重测 R@10 + NDCG). 然后 (2) (3) 是研究型 ROI.

result: Task #267 / Issue #10 — 自主决策 A2 (关 issue 走 NO-GO). 用户 2026-07-29 指示 "do by yourself", R11.3 选 A2 (证据: Sinkhorn 旋钮扫描 / Arm A vs B 碰撞率/R@10 不构成杠杆 / Task #225 §5 同 collision 跨机制 R@10 差 40%). 不启动 4-arm (~13h) / β+Sinkhorn 联合 (~2-4h). 关闭 Issue #10 走 not_planned + 留下建设性产物 (collision_rate 权威定义, 历史数字重述, task200 retro-label). 资源转向 backlog 高 ROI 候选 (paper-aligned R@10 重测).