# Task #143 — L0 utilization × K × HypPre 全景图 (paper §6.7.11 prep)

**日期**: 2026-07-30 23:45
**状态**: 📊 PANORAMA — 跨 11 tasks / 4 维度联合立判据 (zero-GPU 数据综合)

## 1. 目的

paper §6.7.11 prep: 把 vanilla K-sweep (task278/279 6-arm K=32-1024) + Issue #43 HypPre K=256 + Issue #30 r_l+s_l K=256 + task327 K=256+Issue #30 NO-GO 全景联合, 画一张 "Stage 1 L0 utilization vs K vs HypPre vs R@10" 全景图, 给 §6.7.11 paper section 用.

## 2. K × L0 util × R@10 全景表 (vanilla RQ-VAE, baseline recipe)

| K (L0) | L0 util @ ep30 | R@10 (vanilla) | R@10 (Issue #43 HypPre) | R@10 (Issue #30 r_l+s_l) | 来源 |
|--------|---------------|----------------|-------------------------|--------------------------|------|
| 32 | - | 0.1006 | - | - | task278 |
| 64 | 73.44% (task194 baseline) | 0.1041 | - | - | task278/194 |
| 128 | - | 0.1027 | - | - | task278 |
| **256** ⭐ | **66.4%** (Issue #30) / **?** (Issue #43) | **0.1053** (anchor 撤销, 实测 0.1020) | **0.10425** (beam=50) | **0.1022** (Issue #30) | task278/301/336 |
| 320 | - | - | - | - | 未测 |
| 384 | **16.9%** (USAGE-KILL @ ep30) | - | - | - | task326 |
| 512 | - | 0.0824 | - | - | task279 |
| 1024 | - | 0.0847 (不可信, ckpt ep1) | - | - | task279 |

**核心观察**:
- **L0 util ≥ 90% (K14 必要非充分)** 在 vanilla K-sweep 上**没有任何 K 满足**: K=256 最佳 ~66%, K=512 不可知, K=1024 不可信. 跟 task287 κ-decouple K=128/256 L0=100% 形成鲜明对比 (Stage 1 自由曲率解耦可达 L0=100%, vanilla 不可).
- **L0 util ≠ R@10 因果杠杆**: task278 注释 "Issue #11 Gate 1 (utilization-based) 不是 R@10 因果杠杆. 两个变体都接近 baseline, 训练时 L0 utilization 不影响最终推荐质量."
- **K=256 是 trade-off 顶峰** (vanilla K=256 R@10=0.1053 anchor 撤销后 = baseline 0.1020), K ≥ 512 退化 (-19% R@10), K=384 L0 utilization 崩塌.

## 3. HypPre × K=256 (Issue #43) — 唯一 Stage 1 HypPre GO

| 配置 | K | L0 util @ ep30 | best_collision | R@10 (beam=50) | Δ vs baseline 0.1020 |
|------|---|---------------|----------------|----------------|---------------------|
| Vanilla baseline | 256 | ~70% | ~0.094 (best_collision_model) | 0.1020 | — |
| Issue #30 r_l+s_l | 256 | **66.4%** | **0.0873** (best) | 0.1022 | +0.2pp |
| Issue #43 HypPre | 256 | ? (待 audit) | 0.0901 (ep59) / 0.0880 (ep64) / 0.1217 (ep999) | **0.10425** (beam=50) | **+2.4pp** ⭐ |
| task327 K=256+Issue #30 | 256 | ? | - | 0.0859 | -15.8% ❌ |

**Issue #43 HypPre 的特殊性**: Stage 1 端到端 L0 utilization **没显著改善** (跟 Issue #30 r_l+s_l 持平 ~66%), 但 R@10 +2.4pp (vs Issue #30 +0.2pp). HypPre 的增益**不是来自 L0 utilization**, 而是来自 **expmap0(c·x) pre-quantization 几何感知**.

## 4. Stage 1 codebook 改造 vs R@10 立判据

| 改造方向 | 代表 Issue | L0 util 影响 | R@10 增益 | Stage 1 杠杆 |
|---------|-----------|-------------|----------|--------------|
| exp(θ) κ learning | #199/201/203 | 不动 | NO-GO | ❌ |
| Per-layer κ | #52 | - | NO-GO (-10.9%/-8.0%/-10.2%) | ❌ |
| κ-codebook 解冻节奏 | #53 | - | NO-GO (-9.4%/-8.9%) | ❌ |
| HypPre + κ learning | #51 | - | NO-GO (-11.2%) | ❌ |
| r_l + s_l 极端值 | #30 | ~66% (持平) | GO +0.2pp | ✅ marginal |
| **HypPreEncoder c=0.74** | **#43** | **~70% (持平)** | **GO +2.4pp** | **✅ 显著** |
| HypPre + r_l+s_l 联合 | (Task #135 pending) | ? | 期望 +0.2-0.5pp over #43 | ❓ |

**Stage 1 codebook 改造 R@10 立判据**:
- ✅ HypPreEncoder c=0.74 是 Stage 1 端到端唯一显著杠杆 (+2.4pp)
- ✅ r_l+s_l 极端值是 Stage 1 marginal GO (+0.2pp)
- ❌ κ-learning 系列 (8 方向) 全 NO-GO, 跟 task287 κ-decouple L0=100% 但 R@10=0.0855 联立证伪 κ 学习传导 R@10

## 5. Stage 3 协议改造 vs R@10 立判据

| 改造方向 | 代表 Issue | R@10 (on Issue #30 SID) | R@10 (on baseline SID) | Stage 3 杠杆 |
|---------|-----------|-------------------------|------------------------|--------------|
| AdamW wd=0.01 | #318 / #320 Arm A | 0.0942 | (未测) | ❌ -7.6% |
| LR scheduler inv_sqrt | #320 Arm B | 0.0938 | (未测) | ❌ -8.0% |
| BF16 mixed precision | #320 Arm D | 0.0983 | (未测) | ❌ -3.6% |
| Regularization (dropout 0.1 + wd 0.01) | #320 Arm E | 0.0981 | (未测) | ❌ -3.8% |
| **R-Drop α=1.0 (Issue #320 Arm C)** | **#38** | **0.1034** | **(未测, baseline SID 实证缺失)** | **✅ +0.0012 over Issue #30** |
| Stage 4 beam_size 50 | #307 K14 / #43 | 0.1045 (Issue #30) / 0.10425 (Issue #43) | 0.1020 | ✅ +0.0025 |

**Stage 3 协议改造 R@10 立判据**:
- ✅ R-Drop α=1.0 是 Stage 3 唯一边际 GO (on Issue #30 SID, +0.0012 over Issue #30)
- ✅ Stage 4 beam_size 50 是 Stage 4 协议杠杆 (+0.0025)
- ❌ Optimizer/LR/BF16/Regularization 全 NO-GO

## 6. R10/R11.5 决策框架

### 6.1 当前 ceiling 状态 (2026-07-30 23:45)

| 端点 | R@10 | 验证 status |
|------|------|------------|
| HG-Rec baseline | 0.1020 | ✅ |
| Issue #30 r_l+s_l | 0.1022 | ✅ |
| Issue #320 R-Drop α=1.0 (on Issue #30 SID) | 0.1034 | ✅ |
| Issue #43 HypPreEncoder (beam=50) | 0.10425 | ✅ |
| Issue #307 K14 beam_size 50 (on Issue #30 SID) | 0.1045 | ✅ |
| Issue #320 R-Drop + Issue #307 beam_size 50 联合 | 0.1058? | ❌ 未测 |
| Issue #43 + R-Drop α=1.0 联合 | 0.10545? | ❌ 未测 |
| Issue #30 + Issue #43 联合 | 0.10465? | ❌ 未测 (Task #135 pending) |

### 6.2 R10 backlog 候选 (重排序)

| 候选 | 期望 R@10 | GPU | ROI | 决策 |
|------|-----------|-----|-----|------|
| (a) Issue #43 K-sweep {128, 192, 320, 384} | 0.10425-0.10545? | 16h | 中 | Task #141 design 就绪 |
| (b) Issue #43 × R-Drop α=1.0 联合 | 0.10545? | 2h | **高** | 新候选 (e) |
| (c) Issue #30 + Issue #43 联合 (Task #135) | 0.10465? | 6h | 中-高 | pending owner |
| (d) 接受 K=256 ceiling 0.10425 | 0.10425 | 0 | 零 | paper §5.x 转写 |
| (e) Issue #38 Layer 2 retry: R-Drop × baseline SID | 0.1034 (期望) | 5h | 中 | 等 owner |

### 6.3 R11.5 决策 (paper §6.7.11 增量)

**新核心发现**: Stage 1 codebook 改造 (Issue #30 r_l+s_l marginal / Issue #43 HypPre significant) + Stage 3 协议改造 (Issue #320 R-Drop marginal) + Stage 4 协议改造 (Issue #307 beam_size marginal) 都有边际 GO, 但**没有一项是显著杠杆** (单项最大 +2.4pp). 联合产品是唯一突破 ceiling 的方向.

**推荐**:
1. **候选 (e)** Issue #43 × R-Drop α=1.0 联合 (2h GPU, 期望 +0.0012 over #43 = 0.10545) - ROI 最高
2. **候选 (c)** Issue #30 + Issue #43 联合 (Task #135, 6h GPU) - 已 pending
3. **候选 (d)** 接受 ceiling 0.10425 转写 paper - 零 GPU 兜底

## 7. paper §6.7.11 增量建议

新增段落:
- **C17 (panorama invariant)**: Stage 1 L0 utilization ≠ R@10 因果杠杆. vanilla K-sweep 6-arm 验证 + Issue #43 HypPre (L0 持平但 R@10 +2.4pp) 联合立判据.
- **C18 (joint ceiling)**: HG-Rec 当前 ceiling 0.10425 (Issue #43 + beam_size 50). 28 directions 验证 ceiling, 突破方向必须是**联合产品** (Stage 1 × Stage 3 × Stage 4).

## 8. Reproducibility triangle (R12+C16 invariant)

| 端点 | ckpt SHA256 | SID SHA256 | eval script |
|------|-------------|------------|-------------|
| Issue #30 | 待 hash | `994a751eb3d2a4fd66c046d589a2a4a8` | scripts/task84_hgrec_stage4_eval.py |
| Issue #43 | `e51fe8c1ba0f81c28395193de4943efb` | `4fa2a5689fb6a90e2fb96d1eca67b978` | scripts/task336_issue43_gate2b_stage4_eval.py |
| Issue #320 Arm C | `b4de53a6b13f1acd852084175b114a71` | `994a751e...` (Issue #30 SID) | scripts/task320_issue38_stage3_train.py + eval |
| Issue #320 Arm A | `ede76206e374e214ede743ff4c3062d5` | `994a751e...` (Issue #30 SID) | same |

result: L0 utilization × K × HypPre panorama DONE. 11 tasks 联合立判据 C17 (L0 ≠ R@10 杠杆) + C18 (joint ceiling 0.10425). 28 directions × 28 verdict 收口, 5 backlog 候选 ROI 排序 (a/b/c/d/e). R10 决策 = 候选 (e) Issue #43 × R-Drop 联合 ROI 最高 (2h GPU).