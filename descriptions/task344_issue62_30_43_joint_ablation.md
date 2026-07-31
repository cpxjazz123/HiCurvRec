# Task #344 — Issue #62 / #30+#43 联合 ablation (唯一 ROI > 0 路径)

## 来源

承接 Issue #62 (owner 显式 opened, 2026-07-30 16:07):
- A/B/C 三方向 (Issue #55/#56/#57) 全 NO-GO 实证收口
- Issue #57 Stage 4 NO-GO (commit 9fcce34, R15 push)
- 唯一 GO 端点: Issue #30 (R@10=0.1022, +0.2%) + Issue #43 (R@10=0.1041, +2.1%)
- task482 verdict §推荐下一步: "优先启动 Issue #30 跟 Issue #43 联合 ablation"

## 任务

联合 ablation: **Stage 1 RQ-VAE 训练时同时叠加 #43 HypPreEncoder 跟 #30 per-layer Codebook Transforms**, 完整 4 阶段流水线 (Stage 1 + 2 + 3 + 4), 验证联合是否突破 #43 单点天花板.

## Gate 0: 联合 ablation 准备 (zero-GPU)

- 验证 #30 端点 + #43 端点 加载无错
- 设计联合 Stage 1 训练脚本 (HypPreEncoder + Codebook Transforms 组合)
- 单元测试 5/5 PASS (compose regression)

## Gate 1: Stage 1 + Stage 2 重新训练 (~3h/Arm, 4 Arms = ~12h)

| Arm | Stage 1 配置 | Stage 2 SID 文件 |
|-----|--------------|------------------|
| **Arm A** | baseline (HG-Rec Task #84 recipe) | `_t5_hrqvae_poincare.npy` |
| **Arm B** | #43 HypPreEncoder 单独 (c=0.74) | `_t5_hrqvae_hyp_pre.npy` (复用 task336) |
| **Arm C** | #30 单独 (r_l=[0.1,1,10] + s_l=[2,2,2]) | `_t5_rqvae_k0256.npy` (复用 task301) |
| **Arm D** | **#30 + #43 联合 (新)** | (新生成) |

Gate 1 通过条件 (per Arm):
- L0 utilization ≥ 90% @ ep ≥ 50
- L1 utilization ≥ 90% @ ep ≥ 50
- L2 utilization ≥ 90% @ ep ≥ 50
- collision_rate ≤ 0.20

## Gate 2: Stage 3 + Stage 4 eval (~3h/Arm)

| Arm | Stage 3 配置 | Stage 4 R@10 (预计) |
|-----|--------------|---------------------|
| A | T5-mini default | 0.1020 (baseline) |
| B | T5-mini default | 0.1041 (Issue #43 best GO) |
| C | T5-mini default | 0.1022 (Issue #30 marginal GO) |
| D | T5-mini default | (新) |

## 决策矩阵

| Stage 4 R@10 (Arm D) | 决策 |
|----------------------|------|
| **> 0.1042** (+0.5pp) | GO marginal: 联合 ablation 突破单点天花板, 转 [TARGET REACHED] |
| **> 0.1053** (+1pp) | GO 显著: 接近 task194_k0256 仓库最高 |
| **≤ 0.1041** (跟 #43 单点持平) | 联合 NO-GO, 维持 #43 单点 ceiling |
| **< 0.1020** | 联合破坏性 NO-GO (新公式 bug 风险) |

## 实施基础

- **#43 HypPreEncoder wrapper**: `scripts/task334_issue43_gate2a_hyp_pre_encoder.py` (5/5 PASS)
- **#30 Codebook Transforms wrapper**: `scripts/task301_issue30_gate0_codebook_transforms.py` (3/3 PASS)
- **#43 Stage 1 trainer**: `scripts/task336_issue43_gate2b_stage1_train.py`
- **#30 Stage 1 trainer**: `scripts/task301_issue30_gate1_stage1_train.py`
- **#43 端点 SID**: `HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_hyp_pre.npy`
- **#30 端点 SID**: `HG-Rec/dataset/Instruments/Instruments_t5_rqvae_k0256.npy` (symlinked)

## 关键技术决策 (R11.5)

### Joint Stage 1 trainer 设计
1. Build base HRQVAE (HG-Rec standard)
2. Wrap with HRQVAEWithHypPre (Issue #43) — input preprocessing
3. Apply per-layer codebook transforms (Issue #30) — modify `q.embeddings.weight` for each layer
4. Train with same hyperparameters as #30 + #43 (epoch=1000, batch=1024, lr=1e-3, AdamW)

### 兼容性分析
- **#43 HypPreEncoder**: 仅修改输入, 不影响 codebook
- **#30 Codebook Transforms**: 仅修改 codebook, 不影响输入
- **两者正交**: 顺序应用无冲突 (Gate 0 regression test 验证)

### Gate 0 sanity 验证
- Arm D = #43 + #30 同时启用
- 输入: sentence-T5 embedding x (768d)
- 流程: x → HypPreEncoder(expmap0(c=0.74)) → encoder → HRQ codebook (with r_l+s_l transforms) → decoder
- 预期: 训练稳定, L0/L1/L2 util 100%, collision ≤ 0.20

## 总 GPU 时间估算

- Gate 0 (zero-GPU): 30 min (reg test + 设计 + 文档)
- Gate 1 (4 Arms × 3h): 12h (Stage 1 + Stage 2)
- Gate 2 (4 Arms × 3h): 12h (Stage 3 + Stage 4)
- **总 GPU**: ~24h (4 Arms 完整 4 阶段)

> **注意**: Arm A/B/C 是 baseline 对照组, 可复用 #30 #43 现有 Stage 1/3/4 产物 (避免重训). 只有 **Arm D** 需要重新跑完整 4 阶段 (~6h).
>
> **GPU 时间修正**: 6h (仅 Arm D 新训) + ~1h (新 Stage 3 + Stage 4 eval) = **~7h 总 GPU**.

## 关联产物 (预计)

| 类型 | 路径 |
|------|------|
| Joint Stage 1 trainer | `scripts/task344_issue62_joint_stage1_train.py` |
| Joint Stage 1 launcher | `scripts/task344_issue62_joint_stage1_train.sh` |
| Gate 0 sanity test | `scripts/task344_issue62_gate0_sanity.py` |
| Joint Stage 2 SID | `HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_30_43_joint.npy` |
| Joint Stage 1 ckpt | `products/task344/stage1/Instruments/<run_id>/best_loss_model.pth` |
| Joint Stage 3 ckpt | `products/task344/stage3/Instruments/<run_id>/HG_Rec_best.pth` |
| Stage 4 metrics JSON | `verdicts/task344_issue62_stage4_arm{d}_metrics.json` |
| Final verdict | `verdicts/task344_issue62_joint_ablation_verdict.md` |

## R9 顺带修复

descriptions/ 存在 task342/343/345 空洞 (无 scripts/verdicts 关联, 纯 gap). 本任务描述落地后, 配套 task342/343/345 placeholder 同步填补 (R9 enforcement).

---

result: Task #344 / Issue #62 联合 ablation (Issue #30 + #43) = 唯一 ROI > 0 路径. Gate 0 zero-GPU 准备 + Gate 1 Stage 1 重新训练 (Arm D #30+#43 联合, ~3h) + Gate 2 Stage 3+4 eval (~3h). 总 GPU ~6h. 决策: R@10 > 0.1042 = 突破 #43 单点天花板, 转 [TARGET REACHED]. R@10 ≤ 0.1041 = 联合 NO-GO 维持现状.
