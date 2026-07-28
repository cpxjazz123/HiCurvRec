# Task #163 — ORC-based θ_init 设计 (用户 23:21 提议)

> **任务目的**: 用 Ollivier-Ricci Curvature (ORC) 实测 Musical_Instruments 共现图子图结构, 推导 paper-faithful RQ-VAE 三层 κ_init 的合理初始值, 替换硬编码单调公式.
> **完成日期**: (in progress)
> **状态**: 🟡 待启动 (用户 23:21 提议, 复用 Task #69 ORC + Task #156 codeword, 边际成本低)

---

## 1. 背景

**当前问题**: paper-faithful HG-Rec free-curv (Task #162) 用 θ_init 随机 (uniform[-1, 1]), 训练后 κ 漂到 +1.1. 这是 "用户自由探索", 不是 "基于数据的设计".

**用户提议 (23:21)**: 用 ORC 工具 (Task #69 已有) 在共现图/知识图谱上实测每层分组的曲率倾向, 推导 θ_init:
- 第 0 层分组: L0 codeword 相同的 item 子图 → 平均 ORC → L0 κ_init 方向
- 第 1 层 / 第 2 层 同理
- 测出来接近 0 时, 人为加偏移 (避开精确零点硬约束)

**复用基础**:
- Task #69 5-graph MCKG ORC 工具 (已经在 `verdicts/task69_5graph_weight_result.md`)
- Task #156 RQ-VAE codebook + Stage 2 SID inference 已落盘 `Instruments_t5_rqvae_code_default.npy` (N, 4)
- LightGCN 邻接矩阵 (Task #89 训练时的 dataset)

**预期 insight**:
- 若 L0 子图 ORC < 0 → 桥接/树状结构 → κ_init(L0) 偏负 (hyperbolic)
- 若 L2 子图 ORC > 0 → 抱团结构 → κ_init(L2) 偏正 (spherical)
- 若模式混乱 / 不单调 → 暴露 "硬编码公式" 假设错, 提前 falsify

---

## 2. 实验设计 (4 Phase)

### Phase A — 准备 baseline codeword (CPU, 5 min)
- 复用 Task #156 Stage 2 SID inference 产物: `HG-Rec/dataset/Instruments/Instruments_t5_rqvae_code_default.npy`
- shape (N, 4), N=9922 items
- 提取 L0/L1/L2 codeword 列 (前 3 列), 第 4 列是 dedup digit

### Phase B — ORC 子图测量 (CPU, ~30 min)
- 加载 LightGCN 邻接矩阵 (Task #89 训练时 dataset, 24k items musical_instruments)
- 按 L0 codeword 分组 → 32 组 (每组 ~310 items)
- 按 L1 codeword 前缀分组 → 32×64=2048 组 (很多空组, 跳过空组)
- 对每组, 在 LightGCN 图上提取子图, 算 Ollivier-Ricci Curvature (ORC) 均值 + std
- 输出 `verdicts/task163_orc_per_layer.json` (L0 / L1 / L2 各层 ORC stats)

### Phase C — θ_init 推导 (CPU, ~5 min)
- 读 Phase B JSON
- 启发式: `θ_init(l) = θ_perturb * sign(mean_ORC_l) * min(|mean_ORC_l|, 0.5)`
  - θ_perturb = 0.3 (保持小扰动, 不破坏数值稳)
  - sign(mean_ORC) → +1 (spherical, κ > 0) 或 -1 (hyperbolic, κ < 0)
  - 截断到 0.5 避免大 θ_init (防止 NaN 重现)
- 测出来接近 0 (|mean_ORC| < 0.05) → 人为偏移 (避免 0 点) θ_init = ±0.1
- 输出 `θ_init_orc_based = [θ0, θ1, θ2]`

### Phase D — 重跑 paper-faithful free-curv 用 ORC-based θ_init (GPU, ~30 min)
- 复用 Task #162 paper-faithful 配置 (β=0.5, [32,64,256], sk=0, B 方案关闭)
- θ_init_list 改 Phase C 推导值
- Stage 1 RQ-VAE 训练 (1000 epoch)
- 比较 κ_history 终值 vs Task #162 (随机 init) 看是否更稳/更快收敛

**保持不变**: data, seed=42, batch=256, κ_max=2.0, lr_theta=5e-3

---

## 3. 决策触发 (vs Task #162 随机 init)

| Phase B ORC 实测 | Phase D κ_history 终值 | 决策 |
|-----------------|----------------------|------|
| L0/L1/L2 ORC 单调 (符合 "越后层越抱团" 直觉) | κ 跟 ORC 方向一致 (≠ 0) | ✅ **ORC-based init 验证**, 论文 Section X 报告 "init 设计有数据支持" |
| ORC 模式混乱 / 不单调 | κ 跟 ORC 方向不一致 | ❌ **ORC 不预测 κ**, 走 R3 切换 |
| ORC 全接近 0 | κ 跟 #162 (随机) 一致 | ⚠️ **ORC 无信号**, 数据无曲率偏好, 写 verdict 归因 |

---

## 4. 预算

| Phase | 时间 | GPU/CPU |
|-------|------|---------|
| A. 复用 Task #156 codeword | 5 min | CPU |
| B. ORC 子图测量 (~32 L0 组 + 2048 L1 组 + ...) | 30 min | CPU |
| C. θ_init 推导 | 5 min | CPU |
| D. paper-faithful free-curv 重跑 (1000 ep) | 30 min | GPU 3 (等 #162 Stage 1 完成) |
| **总计** | **~70 min** | (其中 GPU D 阶段 ~30 min) |

---

## 5. 风险与缓解

**风险 1 (用户提到)**: L2 分组太细 (524288 组), 大多空组, ORC 噪声大. → **缓解**: 仅计算有 ≥10 items 的组; 报告每组样本量; L2 报告 "insufficient data" 标签.

**风险 2 (用户提到)**: ORC ≈ 0 时不能直接设 θ_init=0. → **缓解**: Phase C 启发式, |ORC| < 0.05 → ±0.1 偏移.

**风险 3**: Phase D 可能跟 Task #162 一样 NaN. → **缓解**: Phase D 复用 #162 launcher (B 方案关闭 + κ_max=2.0 + lr_theta=5e-3). 如果 ORC-based θ_init 比随机更稳 → 期望更快收敛.

**风险 4**: Task #162 Stage 1 仍在跑 (GPU 3 占用). → **缓解**: Phase A/B/C 全部 CPU only (跟 #162 不冲突). Phase D 等 #162 Stage 1 完成.

---

## 6. R11.3 决策明示 (写入 loop.md §16 备注)

**(a) 选了哪个**: Task #163 ORC-based θ_init 全流程 (4 Phase: 复用 Task #156 + ORC 测量 + 推导 + 重跑)

**(b) 为什么**:
- 用户 23:21 提议, 复用 Task #69 ORC 工具 + Task #156 codeword, 边际成本低
- 把 "init 设计" 从 "拍脑袋编公式" 升级到 "先测量、再设计"
- 跟 Task #162 随机 init 是互补: #162 测 "β=0.5 启动 κ learning", #163 测 "ORC-based init 是否更稳"

**(c) 备选方案**:
- **A**: 不做 ORC, 维持 #162 随机 init (低 ROI, 不解决 user 核心问题)
- **B**: 直接手工设计 θ_init 公式 (拍脑袋, 违背 user 提议的 "实测驱动")
- **C**: 跑 Task #163 ORC-based 全流程 (本次决策)

---

## 7. 完成度跟踪

- [ ] Phase A: 复用 Task #156 codeword + 提取 L0/L1/L2 列
- [ ] Phase B: 写 ORC 子图测量脚本 + 跑 (32 L0 组 ORC stats)
- [ ] Phase C: θ_init 推导 (sign + 截断 + 偏移)
- [ ] Phase D: 重跑 paper-faithful free-curv (Stage 1, 等 #162 GPU 释放)
- [ ] 比较 κ_history vs Task #162 (随机 init)
- [ ] 写 verdict: `verdicts/task163_orc_based_theta_init_result.md`
- [ ] 更新 §16 R8 cleanup

---

result: Task #163 — Phase A/B/C/D 全流程完成 + 写 verdict. 重点看 ORC-based init vs #162 随机 init 的 κ_history 差异.