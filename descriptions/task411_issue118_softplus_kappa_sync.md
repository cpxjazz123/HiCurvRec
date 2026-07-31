# Task #411 / Issue #118 [方向A Gate1] 非零曲率参数化 + 更新后同步重标定

**日期**: 2026-07-31
**Issue**: #118 [方向A Gate1] — 针对 #115 κ_m 卡 0 死区, 改用 κ_l=-softplus(u_l)-ε + 每次 κ_l 更新后按 #47 同步重标定 codebook/distance
**任务**:
1. **Real-entry 因果断言**: 固定 batch, 改变 u_l 必须改变 κ_l/scale/distance, 梯度有限非零 (断言通过 → 才允许训练)
2. **主配置**: FreeCurvVQ K=[64,128,256] 30 epoch + softplus κ_l + 同步重标定
3. **Control**: 旧 tanh 参数化 (跟 #115 一致)
4. **每 epoch 导出**: κ, dκ/du, codebook norm, util, max_load, entropy, loss, NaN/Inf, 更新前后一致性残差
5. **PASS**: 三层 util≥90%, max_load<5%, κ/u 梯度有限非零并离开初始化, norm 不退化, 同步残差 in tolerance, 无 NaN/Inf

**前置** (per spec §Framework compliance precheck):
- 保留 L0 K64 / L1 K128 / L2 K256 三层独立 learnable variable-curvature 主路
- 每层独立 u/κ/scale/codebook
- 禁止 global κ, fixed-only, pure Euclidean bypass, 随机/hash SID

**结果**: ⏸ 进行中 (R22 + R19 立即开工)

---

## R17 + R20 强制 4 Gate 详细内容

### Gate 1 (= Stage 1 softplus κ_l + sync rescale): ⏸ 进行中

**两阶段实施**:
- **Phase 0**: 真实 batch 因果断言 (微扰 u_l → κ/scale/distance 改变 + gradient 有限非零)
- **Phase 1**: 主配置 30 epoch softplus κ_l + 同步重标定
- **Phase 2**: Control 30 epoch 旧 tanh 参数化 (跟 #115 一致)

**PASS 阈值**:
- 三层 utilization ≥ 90%
- max_load < 5%
- 每层 κ/u 梯度有限非零并离开初始化
- norm 不退化, 报告预注册下界
- 同步残差 in 声明容差
- 无 NaN/Inf
- 提交 ckpt/SHA256/log/verdict/commit

### Gate 2/3/4: ⏸ STOP per spec
- 原因: Gate 1 必须先 PASS

---

## 跨方向联立 (R18 实证 4 维度)

| 维度 | Issue #115 (task408, closed) | Issue #118 (本 task) |
|------|------------------------------|----------------------|
| **D1 spec 摘录** | θ_m init=0 + tanh(θ) 卡 0 死区 | **softplus(u_l)+ε 替代 tanh, 非零曲率参数化** |
| **D2 实施核心** | FreeCurvVQ K[64,128,256] 30 epoch | **同步重标定 codebook/distance (per #47 公式)** |
| **D3 Gate 失败机制** | κ_m 卡 0, L0 util 75% | **softplus 强制 κ>0, 同步重标定防 norm 退化** |
| **D4 引用文献** | arXiv:2405.13979 | arXiv:2405.13979 ✅ 同文献 |

**R18 v2 强制结论**: Issue #118 跟 #115 路径**有差异** (参数化方式 + 同步重标定). 必须做新实验.

---

## R11.5 自主决策记录

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 参数化 | κ_l = -softplus(u_l) - ε (ε=1e-3) | Issue spec 强制要求 (替换 tanh 卡 0 死区) |
| 同步重标定 | 按 #47 公式, 每次 κ_l 更新后 rescale codebook | Issue spec 强制要求 |
| K 配置 | K=[64,128,256] (跟 #115 一致) | Issue spec §Framework compliance precheck 固定 |
| 训练 epoch | 30 (主配置) + 30 (control) = 60 epoch total | Issue spec "只跑主配置和 control" |
| Sinkhorn | sk_eps=0.003 + sk_iters=3 (跟 HG-Rec baseline) | Issue spec 强制 |
| GPU 分配 | GPU 3 (per R7 全部空闲) | R7 + R19 跨 issue 并行 (Issue #119 占 GPU 2) |
| Conda env | `genrec_env` (R1 默认) | R1 v1 通用 GPU/CPU 计算任务 |

---

## 后续 (per R22 + R19 + R16)

1. ✅ 写 description (本文件)
2. ⏳ 写 softplus κ_l + sync rescale 训练脚本
3. ⏳ Gate 1 因果断言 + 训练 60 epoch
4. ⏳ 写 Gate 1 verdict
5. ⏳ commit + push (R15 + R21 v2)
6. ⏳ close Issue #118 (R16 + R20 + R21)