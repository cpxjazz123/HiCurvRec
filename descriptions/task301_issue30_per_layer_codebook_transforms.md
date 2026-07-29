# Task #301 / Issue #30 — per-layer 异构 Codebook Transforms (r_l + R_l + s_l + per-layer c_k range)

**日期**: 2026-07-29
**状态**: Gate 0 PASS, Gate 1 启动
**承接**: Issue #30 (AI 自主决策 — 9 方向 NO-GO + #28 + #29 启动后第 12 方向 per-layer 异构码本几何)
**关联**: [[issue30-task-body]] [[task298-issue28-result]] [[task297-issue25-result]] [[task290-fsq-kappa-decouple-result]]

---

## 1. 任务定义

**目的**: per-layer 异构码本几何 (per-layer r_l radius + per-layer R_l rotation + per-layer s_l scale factor) + per-layer c_k range, 在 baseline Stage 1 recipe 之外架构层推进 per-layer 可变"码本空间"机制.

**触发条件**:
- 9 方向 × 14 verdict 全 NO-GO 收口 (task294 + task296 + task297)
- Issue #28 启动 Gumbel-Softmax 路径 + Issue #29 启动 K_l 路径 (互补)
- Issue #30 是 9 方向 + 3 架构层方向 = 第 12 方向
- owner feedback 2026-07-29 23:13「不允许假设 owner 有 decision. 每次 loop. AI 必须自行决策做出可以推进目的的决定」

**否决假设**:
- 不重复 FSQ 全层同构 fixed bins (task290 -45.8% 已 NO-GO)
- 不重复 EMA / Restoration (task291 / task292 -25.0% / -21.7% NO-GO)
- 不重复 per-codeword κ (task231 Stage 1 坍缩)
- 不修改 HG-Rec/model/ 上游源码 (R11.4 critical decision)
- 不跳过 4-Gate 硬停止

---

## 2. 4-Gate 硬停止协议

**Gate 0 —— 实现 per-layer 异构 Codebook Transforms 训练代码**

- 实现 `train_hrqvae_codebook_transforms.py` 继承 baseline
- 新增 per-layer `radius_list` / `rotation_list` / `scale_list` 参数
- 在 baseline HRQVAE 构造后, 对每层 HVectorQuantization.embeddings.weight 应用 per-layer 几何变换:
  ```python
  e_i^l → (s_l · r_l) · R_l · e_i^l  (切空间)
  ```
- 回归测试: r_l=[1,1,1] / R_l=I / s_l=[1,1,1] 输入下 forward 与 baseline 完全一致 (max diff = 0)
- **硬停止**: Gate 0 FAIL → STOP, 不进入 Gate 1

**Gate 1 —— Stage 1 100 epoch 训练**

- 端到端 Stage 1 训练 (transformation 在 init 时一次应用, 后续优化器直接更新变换后的码本)
- per-layer r_l = [0.1, 1.0, 10.0] (L0 紧凑, L1 中等, L2 宽松)
- per-layer s_l = [2.0, 2.0, 2.0] (per-layer 异构 scale factor)
- per-layer R_l = I identity (本 issue 不引入非平凡 rotation, 后续 task 可调)
- per-layer c_k_range_list = [(1.0, 5.0), (0.5, 20.0), (0.5, 20.0)] (task242 Arm A)
- L0/L1/L2 utilization ≥ 90% + collision_rate ≤ 0.20
- **硬停止**: 任一不满足 → STOP, 不进入 Gate 2

**Gate 2 —— Sinkhorn 5 iter 推断**

- 用 Gate 1 best ckpt 跑 Sinkhorn max_iters=5
- 4-digit SID unique count ≥ 9500 + per-layer util 偏差 ≤ 5pp
- **硬停止**: unique < 9500 → STOP, 不进入 Gate 3

**Gate 3 —— T5-mini 200 epoch 训练 + Stage 4 评估**

- Stage 4 eval: Test R@10 > 0.1020
- **硬停止**: R@10 ≤ 0.1020 → STOP, 不允许"继续 r_l 的下一变体"

---

## 3. 关键决策点 (R11.3)

| # | 决策 | 选择 | 备选 | 理由 |
|---|------|------|------|------|
| 1 | 不修改 HG-Rec/model/ | ✅ 在 baseline HRQVAE 构造后 init 时应用 transformation | patch HVectorQuantization.forward | R11.4 critical decision, 不动上游 |
| 2 | transformation 应用时机 | ✅ init 时一次应用 + 优化器直接更新变换后码本 | per-forward monkey-patch | 简单 + 快 + 优化器路径清晰 |
| 3 | per-layer r_l 默认 | ✅ [0.1, 1.0, 10.0] (差异显著的 3 个值) | Issue #30 body [0.5, 1.0, 2.0] | 验证 transform 显著生效, 跟 Gate 0 验证一致 |
| 4 | per-layer R_l 默认 | ✅ I identity | 随机 rotation matrix | Issue #30 body 明确, 后续可调 |
| 5 | per-layer s_l 默认 | ✅ [2.0, 2.0, 2.0] | [1.0, 1.0, 1.0] (退化为 r-only) | 跟 r_l 配合放大差异 |

---

## 4. 物理产物

- `scripts/task301_issue30_gate0_codebook_transforms.py` (360 行, Gate 0 验证 + 回归测试 PASS)
- `scripts/task301_issue30_gate1_stage1_train.py` (Stage 1 训练 wrapper, 215 行)
- `scripts/task301_issue30_gate1_stage1_train.sh` (Gate 1 100 epoch 训练 launcher, GPU 1)
- `verdicts/task301_issue30_gate0_verify.json` (机器可读 verify 结果)

---

## 5. Gate 0 验证结果 (2026-07-29)

| 验证项 | 实测 | 决策 |
|-------|------|------|
| 回归测试 identity transforms (r=[1,1,1]/R=I/s=[1,1,1]) | max \|diff\| = 0.00e+00 | ✅ PASS |
| Issue #30 design (r=[0.1,1,10]/R=I/s=[2,2,2]) vs baseline | mean \|diff\| = 6.74e-03 | ✅ PASS (新设计, 非零) |
| Shape 一致性 | baseline / regression / design 都是 [4, 768] | ✅ PASS |
| monkey-patch 干净恢复 (Issue #30 Gate 0 用 monkey-patch 验证) | 两次 forward max diff = 0 | ✅ PASS |

**Gate 0 通过决策**: ✅ 进入 Gate 1 (Stage 1 100 epoch GPU 1)

---

result: Task #301 / Issue #30 Gate 0 PASS. per-layer 异构 r_l=[0.1,1.0,10.0]/R=I/s=[2,2,2] + per-layer c_k range 上游 HRQVAE 已支持 (无需修改 HG-Rec/model/). 回归测试 baseline vs identity transforms max diff = 0.00e+00 (完全相同). Issue #30 design vs baseline mean diff = 6.74e-03 (新设计差异). Shape 一致. Gate 1 Stage 1 100 epoch launcher 已就位 (GPU 1).
