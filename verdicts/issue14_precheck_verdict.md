# Issue #14 Precheck — R18 4 维度对比

**Issue #14**: [方向A] Gate4 单seed Task84 全量评估 — beam K=20 六指标产出 + kappa/codebook 同步性取证

**Status**: precheck ✅ PASS (per #14 顶部 4 项 protocol + Stage 3 val 评估表)

---

## R18 4 维度对比 (vs Issue #13 [方向B], 同维度平行)

| 维度 | Issue #13 (方向B, 已关 061e4cc) | Issue #14 (方向A, 新) | 差异 |
|---|---|---|---|
| **D1 spec 摘录** | Step 1-6 + 单 seed 全 eval (Step 7 待派工) | 单 seed 全 eval + κ/codebook sync 取证 | **D1 差异**: #14 范围更窄 (单步评估 + sync 取证), #13 含 audit + 混合分量诊断 |
| **D2 实施核心** | 4 个 GPU 跑 4 步 (canary_argmax / canary_fix / mixing_diag / beam_search) | 全量 24772 samples × beam K=20 + κ/codebook 数值同步验证 | **D2 差异**: #14 评估规模大, 单跑需要 ~108 min (0.26s/sample × 24772), 多卡并行 ~27 min |
| **D3 Gate 1 失败机制** | taskB #13 Step 1 baseline FAIL, Step 2-3 修复 PASS | 不适用 (评估任务, 跟修复并行, 修复由 #12 已完成) | **D3 差异**: #14 假设 #12 修复已稳态, 不再 FAIL |
| **D4 引用文献** | arXiv + DOI 多种 (SID design + 乘积流形) | 5 篇 arXiv (SID design + test-time scaling) + CrossRef/PubMed 排除 | **D4 差异**: #14 多了 Gryphon / UniRec / PROMISE 等 test-time scaling 工作 |

**D1+D2+D4 都不同**. R18 强制: **必须做实验** (评估 + sync 取证, 不能只落 verdict 文字).

---

## 跟 #12 [方向A] 关系

#12 (commit 469c99e + 380228f) Step 1-4 已完成:
- Step 1: canary 100 samples @ taskA_stage3_issue192_long_run/best_adapter.pt (epoch 48) → R@10=0.01, validity=100%, 4 项 protocol PASS
- Step 2: P2 vocab mapping hash aacb3085
- Step 3: P4 修复 (taskA_stage4_resume.py main flow 用 autoregressive_predict)
- Step 4: beam search K=20, 100 samples → R@5=0.04, R@10=0.05, R@20=0.09, NDCG@5/10/20=0.0195/0.0226/0.0332, 6/6 unique

#14 = #12 Step 5 (单 seed 全 eval 待派工) — 跨 issue 平行, 跟 #13 Step 7 平行.

---

## 实施计划 (3 步)

### Step 1: 单 seed Task84 test 全量评估
- 写 `taskA/stage4/taskA_stage4_full_eval_issue14.py` (跟 #12 Step 4 平行, 但 24772 samples 全量)
- 4 卡并行 split: GPU 0/1/2/3 各跑 6193 samples (~27 min/卡, 总 ~30 min wall)
- nohup background launch
- 输出 6 指标 (R@5/10/20, NDCG@5/10/20) + validity=100% + protocol_audit 4 项 + elapsed + seed
- 落 `verdicts/issue14_full_eval_result.json` (聚合 4 卡) + 各卡 partial json

### Step 2: κ/codebook 同步性复核 (方向A 特有)
- 写 `taskA/stage4/taskA_stage4_kappa_codebook_sync_issue14.py`
- 加载 taskA_stage3_issue192_long_run/best_adapter.pt
- 加载 Stage 2 hrqvae (per-layer learnable κ 在 taskA_stage2_kappa_sync.py L111 + taskA_stage2_kappa_vq_fix.py L104)
- 数值验证: 更新 κ 前后 codebook 距离矩阵 (基于 poincare_distance c=κ) 确有变化
- 文件:行号 + 数值前后对比
- 落 `verdicts/issue14_kappa_codebook_sync.json`

### Step 3: R@10 ≤ 0.1020 分类 (如适用)
- (a) 映射/约束层残留不一致 — check P1/P2/P3/P4 audit 全部 PASS?
- (b) BoundedKappaScaleConditioner `softplus(alpha_logit).clamp(max=0.5)` 双重削弱 κ 信号 — 打印 alpha_value, 看是否接近 0.5 上限
- (c) Stage3 适配器容量不足 — 看 conditioner 梯度 norm vs 总参数占比
- 三类根因各自可核实依据

---

## 跟 #13 关系

#14 是 #13 Step 7 的方向A 平行任务. 都属于"单 seed Task84 全 eval" 范畴:
- #13 Step 7 待 owner 派工 (issue 关闭时已标记)
- #14 即对应方向A 这部分派工 (owner 在 #14 描述里指明: "单seed 真实 R@10 超过 0.1020 之前, Gate4 验收只需一个可复现的单seed 正式 run")

跨 issue 跨 GPU 占用 (R7):
- taskA (方向A): GPU 0/1 跑 #14 full eval
- taskB (方向B): GPU 2/3 待 #13 Step 7 派工时启动
- 4 张 L40S (46GB/卡) 全空, 互不挤占

---

## precheck 状态

| 项 | 状态 |
|---|---|
| precheck (Stage 1+2+3) | ✅ PASS (per #14 顶部 4 项 protocol + Stage 3 val 评估表, val_R@10 @ epoch 48 = 0.058) |
| Stage 4 decode 修复 | ✅ PASS (per #12 Step 2: taskA_stage4_resume.py 用 autoregressive_predict, validity 25% → 100%) |
| full eval 脚本 | ⏳ 待写 (本 tick Step 1) |
| κ/codebook sync 证据 | ⏳ 待产 (本 tick Step 2) |
| Gate 1 状态 | ✅ PASS (verdicts/gate1_evidence.json, sid_sha256=2dab29) |
| Gate 2 状态 | ✅ PASS (per-layer κ 梯度非零, reload SID 5/5 一致) |
| Gate 3 状态 | ✅ PASS (#12 Step 1-4 + #14 顶部 4 项 protocol 验证) |
| Gate 4 状态 | ⏳ conditional (full eval + sync 取证后定论, 本 tick 实施) |

**结论**: 4 维度对比 + 跟 #12 #13 关系 + 实施计划明确, **precheck 通过**. R19 立即启动 Step 1 full eval (本 tick, 4 卡 nohup), Step 2 sync + Step 3 分类后续 tick.

---

## 4 张 GPU 占用计划

| GPU | 任务 | 期望耗时 |
|---|---|---|
| 0 | #14 Step 1 full eval chunk 0/4 (samples 0-6193) | ~27 min |
| 1 | #14 Step 1 full eval chunk 1/4 (samples 6193-12386) | ~27 min |
| 2 | #14 Step 1 full eval chunk 2/4 (samples 12386-18579) | ~27 min |
| 3 | #14 Step 1 full eval chunk 3/4 (samples 18579-24772) | ~27 min |

聚合: 4 partial json → 1 final verdict json
总 wall time: ~30 min (含启动 + 落盘)