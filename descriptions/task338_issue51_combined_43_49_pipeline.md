# Task #338 — Issue #51 — #43 预量化双曲映射 + #49 per-layer 可学习曲率组合实验

**状态**: 待启动 (2026-07-30)
**Issue**: https://github.com/WENYULIANG123/GeneRec/issues/51

## 目的

拆开 Issue #49 "传输损耗" 失败模式的两个混杂因素:
- **H1 (信号匮乏假设)**: #49 失败是因为它操作的残差信号已经被编码器压平 (Task #80), 可学习曲率没有真实双曲结构可学。
- **H2 (架构性传输损耗假设)**: 量化器内部几何正确性本身传不到 R@10, 即便残差信号质量更好也无用。

组合测试 #43 的 HypPreEncoder (c=0.74 固定预映射, 让更多双曲结构存活到残差空间) + #49 的 per-layer 可学习 κ (Issue #47 修复公式 + 解耦调度) 是**唯一能拆开这两个混杂因素**的实验。

## 已知依赖

| 组件 | 来源 | 状态 |
|------|------|------|
| HypPreEncoder (c=0.74 fixed) | Issue #43 / Task #334 (`scripts/task334_issue43_gate2a_hyp_pre_encoder.py`) | ✅ 已验证 PASS R@10=0.1041 (+2.1%) |
| 统一 κ-stereographic 公式 | Issue #47 / Task #338 (`scripts/task338_issue47_fix_unified_formula.py`) | ✅ 5/5 测试 ALL PASS |
| FreeCurvHRQVAE + 解耦调度 (Phase A+B) | Issue #49 / Task #340 (`scripts/task340_issue49_stage1_train.py`) | ✅ Arm B 跑通, 学到 κ=[-0.128,-0.110,-0.123] |
| Sinkhorn 维持 codebook 健康 | Issue #49 验证有效的部分 | ✅ sk_eps=[0.01,0.01,0.01] |
| 真实数据噪声底线 | Issue #50 / Task #337 | ✅ L0 p95=0.21, δ=0.02 校准 |

## 实验设计

**架构组合**:
```
x (768d raw)
  ↓ HypPreEncoder (expmap0, c=0.74)        ← Issue #43 组件 (fixed κ)
↓
encoder (MLP)
  ↓ z (32d)
FreeCurvHRQVAE.hrq (per-layer learned κ)   ← Issue #49 组件 (Issue #47 修复公式)
  ↓ Sinkhorn SID (3-digit + 4th digit dedup)
↓
T5-mini → R@10
```

**Stage 1 训练** (复用 Issue #49 Arm B Phase A+B 配置):
- θ_init = -0.02 (Issue #49 学到有意义 κ 值的臂)
- Phase A (ep 1-200): θ frozen, codebook 训练, Sinkhorn essential (sk_eps=[0.01,0.01,0.01])
- Phase B (ep 201-400): θ unfrozen lr=1e-5, 学到 κ
- Recipe: num_emb_list=[64,128,256], e_dim=32, layers=[512,256,128,64]
- + **新增**: HypPreEncoder(c=0.74, enabled=True) wrapped around FreeCurvHRQVAE

**Stage 1 Gate 1 标准** (与 #49 一致):
- L0/L1/L2 utilization ≥ 90%
- three-digit collision_rate ≤ 0.20
- κ_m 学到非饱和 (|κ| ≪ κ_max=2.0) 非退化 (|κ| > 0.01) 值
- **额外**: 叠加 HypPreEncoder 不应破坏 #49 已验证的 codebook 健康

**Stage 2-4** (与 #49 一致):
- Stage 2: Sinkhorn 推断 + 4-digit dedup → (9922, 4) SID
- Stage 3: T5-mini 200 epoch (复用 task84 训练脚本, 加 `--r12_force_save`)
- Stage 4: test eval beam=20 + beam=50

## 决策阈值 (vs HG-Rec baseline 0.1020 + Issue #43 单跑 0.1041)

| 条件 | 含义 | 决策 |
|------|------|------|
| **R@10 > 0.1041** | 显著超过 Issue #43 单独结果 (≥+0.2pp) | **H1 成立**, 信号匮乏是 #49 失败原因, 组合方向有继续投入价值 |
| **0.1020 < R@10 ≤ 0.1041** | 中性偏 H2 | H2 弱成立, #43 固定映射已吃掉大部分收益 |
| **R@10 ≤ 0.1020** | 叠加可学习曲率破坏 #43 效果 | **H2 强成立**, 应建议后续只用 #43, 不再叠量化器内部曲率学习 |

## 风险与不确定性

1. **架构冲突风险**: HypPreEncoder 改变输入分布, FreeCurvHRQVAE 的 θ_init=-0.02 可能不再是健康起点, 需要重新调参。Gate 1 不通过本身也是有价值信号。
2. **时间预算**: 4 阶段 ~5h total (Stage 1 ~3h Phase A+B + Stage 2 5min + Stage 3 ~2h T5 + Stage 4 5min)
3. **GPU 占用**: GPU 1 (Issue #51) - GPU 0/2/3 空闲

## 产物

| 阶段 | 路径 |
|------|------|
| Stage 1 训练 | `products/task338/{phase_a_final,best_collision_model,phase_b_final}.pth` |
| Stage 2 SID | `products/task338/sid_*.npy` |
| Stage 3 T5 | `products/task338/t5_*/` |
| Stage 4 评估 | `verdicts/task338_issue51_stage4_beam{20,50}.json` |
| Verdict | `verdicts/task338_issue51_result.md` |
| Description | `descriptions/task338_issue51_combined_43_49_pipeline.md` |

## R11.5 自主决策记录

- 启动决策: R10 主动推进模式 + owner 显式 issue #51 派工, 双重授权
- 范围选择: 1 臂 (Arm B θ=-0.02, Issue #49 唯一通过 Gate 1 的臂) — 不做 3 臂, 避免 R9 drift-cycle 风险
- Stage 3 T5 epoch: 200 (Issue #49/Issue #43 默认)
- Sinkhorn: sk_eps=[0.01,0.01,0.01] (Issue #49 验证有效的 essential 值)
- beam_size: 20 + 50 (双测试, 跟 Issue #43 Stage 4 一致)