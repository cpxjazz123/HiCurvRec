# Task #412 / Issue #119 [方向B Gate1] 混合距离评分中的 gate 函数依赖与因果梯度准入

**日期**: 2026-07-31
**Issue**: #119 [方向B Gate1] — 混合距离评分 d_mix=Σ_j α_lj d_lj, gate 直接参与距离计算 (针对 #116 gate 零梯度)
**任务**:
1. **Real-entry 因果断言**: 固定 batch, 微扰单层 gate_logits 必须改变 d_mix/assignment score/loss, autograd gate gradient 有限非零, 置换 weight 必须改变 score (任一失败即 FAIL, 禁止训练)
2. **主配置**: ThreeComponentHRQVAE K=[64,128,256] 30 epoch, 训练期 d_mix 可微, 导出 SID 保持 hard argmin
3. **Control**: uniform-gate control
4. **每 epoch 记录**: component distance, gate weights/entropy/grad norm, utilization, max_load, codebook norm, loss, NaN/Inf, save/load round-trip
5. **PASS**: 三层 util≥90%, max_load<5%, 每层至少 2 component weight>0.1, gate 对 d_mix/loss 梯度有限非零且偏离初始化, L1/L2 norm≥0.2×L0, round-trip in tolerance, 无 NaN/Inf

**前置** (per spec §Framework compliance precheck):
- L0 K64 / L1 K128 / L2 K256 三层独立 learnable-κ hyperbolic 主路
- 每层 + fixed-hyperbolic + Euclidean component + 独立 gate
- 禁止 global κ, 删除 learnable κ, fixed-only, pure Euclidean bypass, 独立模型
- save/load schema/forward 一致

**结果**: ⏸ 进行中 (R22 + R19 立即开工)

---

## R17 + R20 强制 4 Gate 详细内容

### Gate 1 (= Stage 1 因果断言 + d_mix 训练): ⏸ 进行中

**两阶段实施**:
- **Phase 0**: 真实 batch 因果断言 (微扰 gate_logits → d_mix + assignment + loss 都改变 + gradient 有限非零 + 置换 weight → score 改变)
- **Phase 1**: 主配置 30 epoch + d_mix 可微 + hard argmin 导出 SID
- **Phase 2**: Control uniform-gate (gate_logits = 0)

**PASS 阈值**:
- 三层 utilization ≥ 90%
- max_load < 5%
- 每层 ≥ 2 component weight > 0.1
- gate 对 d_mix/loss 梯度有限非零且偏离初始化
- L1/L2 norm ≥ 0.2 × L0
- round-trip in 预声明容差
- 无 NaN/Inf
- 提交 ckpt/SHA256/log/verdict/commit

### Gate 2/3/4: ⏸ STOP per spec
- 原因: Gate 1 必须先 PASS

---

## 跨方向联立 (R18 实证 4 维度)

| 维度 | Issue #113 (task407, closed) | Issue #116 (task409, closed) | Issue #119 (本 task) |
|------|------------------------------|------------------------------|----------------------|
| **D1 spec 摘录** | 三分量 schema (R137 fix) | 连续 distortion + 冻结 | **d_mix=Σ_j α_lj d_lj 让 gate 进入距离** ✅ |
| **D2 实施核心** | 单阶段 hard assignment | 两阶段交替 (gate + codebook) | **soft distance + d_mix 训练期可微** ✅ 新路径 |
| **D3 Gate 失败机制** | 4 轮 v1-v4, gate stuck init | gate_logits 零梯度 + codebook 随机坍缩 | **待实证: d_mix 让 gradient 流到 gate** ✅ |
| **D4 引用文献** | arXiv:2307.04514 | arXiv:2307.04514 | arXiv:2307.04514 WEIGHTED-PM ✅ |

**R18 v2 强制结论**: Issue #119 跟 #113/#116 路径**有差异** (d_mix 让 gate 真的进入 forward distance 计算). 必须做新实验.

**关键差异 vs #113/#116**:
- #113 gate 没参与 forward (跟 hard argmin 解耦)
- #116 gate 也不参与 forward (只在连续 distortion loss 里)
- #119 gate 直接参与 d_mix (跟 #28/#33 区分: #28 Gumbel 替换 hard assignment + 温度 sweep, #33 per-item soft STE; #119 不替换 hard assignment, 不扫温度)

---

## R11.5 自主决策记录

| 决策点 | 选择 | 理由 |
|--------|------|------|
| d_mix 公式 | d_mix = Σ_j α_lj d_lj, 训练期 soft + 推断 hard | Issue spec §Gate1 强制 |
| 训练期 distance | soft distance via softmax(-d_mix) weighted expectation | 让 gradient 流到 gate_logits |
| 推断期 distance | hard argmin (per Issue spec "导出 SID 仍为 hard argmin") | Issue spec §Gate1 强制 |
| 因果断言 | 微扰 gate_logits + soft distance gradient check | Issue spec §Gate1 1 强制 |
| K 配置 | K=[64,128,256] | Issue spec §Framework compliance 固定 |
| 训练 epoch | 30 主配置 + 30 control | Issue spec "仅主配置与 uniform-gate control" |
| GPU 分配 | GPU 2 (per R7 全部空闲) | R7 + R19 跨 issue 并行 (Issue #120 占 GPU 0) |
| Conda env | `genrec_env` (R1 默认) | R1 v1 通用 GPU/CPU 计算任务 |

---

## 后续 (per R22 + R19 + R16)

1. ✅ 写 description (本文件)
2. ⏳ 写 d_mix gate-dependent distance 训练脚本
3. ⏳ Gate 1 因果断言 + 训练 60 epoch
4. ⏳ 写 Gate 1 verdict
5. ⏳ commit + push (R15 + R21 v2)
6. ⏳ close Issue #119 (R16 + R20 + R21)