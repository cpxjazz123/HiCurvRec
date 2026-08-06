---
task: 177
type: verdict
created: 2026-08-02
tags:
  - baseline
up: "[[index]]"
---
# Task #177 β(x) = β_base / λ_κ(x) conformal factor — Broken Baseline 终止

> **任务目的**: 在 Stage 1 RQ-VAE 的 commitment loss 上叠加 conformal 位置依赖 β(x) = β_base / λ_κ(x), λ_κ = 2/(1-c·‖x‖²) 是 Poincaré 共形因子, 验证是否能提升 Musical_Instruments 上 Stage 3 的 R@10 (基线 HG-Rec #84 = 0.1020).

> **完成日期**: 2026-07-25
> **状态**: 🛑 Broken Baseline 终止 (Phase 0 修复后, 已在 GPU 2 上终止 PID 343387)

---

## 1. 执行时间线

| 阶段 | 时间 | 状态 |
|------|------|------|
| Stage 1 RQ-VAE 训练 (β conformal 位置依赖) | 2026-07-25 13:37 | ✅ 完成 |
| Stage 2 SID codebook inference | 2026-07-25 13:44 | ✅ 完成, `_t5_rqvae_posdep_conformal.npy` 落盘 |
| Stage 3 T5-small 5.5M 训练 (PID 343387) | 2026-07-25 14:12 起 | ⏸️ 进行到 epoch 9/200 (~30 min) 时被终止 |
| Stage 4 eval | - | ❌ 未启动 |

**Stage 3 训练进程** 在 broken baseline 上跑了约 30 min, 完成 9 epoch / 200 (early_stop=20 patience 未触发). 在 Phase 0.1-0.3 修复了 HG-Rec 量化器 6 处代码-论文不匹配之后, 决定终止并归类为 broken baseline 历史.

## 2. 关键指标

**Stage 1 + 2 产物** (在 broken baseline 上产出):
- `dataset/Instruments/Instruments_t5_rqvae_posdep_conformal.npy`: shape `(9922, 4)` int, 跟其他 broken-baseline SID 一并归档
- 保留作为历史参考, 不进入未来 Stage 4 评估

## 3. 分析解读

### 3.1 Broken baseline 范围

跟 Task #176 同一套 broken 代码 (用户 2026-07-25 确认, Phase 0 验证):

- β 挂反 (van den Oord 标准约定违背)
- loss 输入未 expmap0 (球外点代入 poincare_distance → artanh 饱和)
- 多余 logmap0 + 3 行死代码
- 默认值 `[32,64,256]` β=0.25 (论文 Table 6 是 `[64,128,256]` β=0.5)
- get_codebook 跟 forward 不一致

**结论**: Task #177 的 Stage 1 训练 + Stage 2 推断产物建立在有倾向坍缩结构的 broken code 上. 即使训完 Stage 3, R@10 也不能跟未来 fixed-baseline 对比.

### 3.2 Conformal β 设计 vs 修复

Task #177 的 conformal β(x) = β_base / λ_κ(x) 是基于"在 Poincaré 流形上, 共形因子 λ_κ 反映位置依赖"的设计. 但 broken baseline 的关键 bug 之一就是"loss 输入未 expmap0 到流形上, 直接代入 poincare_distance". 这意味着:
- Task #177 的 conformal 设计在 broken baseline 上, 实际 λ_κ 计算的是原始欧式坐标的范数, 不是真正的双曲范数
- 修复后 conformal 公式仍然有效 (输入先 expmap0), 但语义跟原始设计略有差异

## 4. 产物清单

| 路径 | 说明 | 状态 |
|------|------|------|
| `products/task177/_TRAINING_PID` | 训练 PID 文件 | 🗑️ 已清 |
| `products/task177/_STAGE4_TRIGGER.sh` | Stage 4 自动触发脚本 | 🗑️ 已清 |
| `products/task177/posdep_conformal/` | Stage 1 ckpt | 🗄️ 归档保留 (broken baseline 标签) |
| `products/task177/t5small_posdep_conformal/jul-25-2026_14-12-06/Instruments/Jul-25-2026_14-12-55/` | Stage 3 T5 ckpt | 🗄️ 空目录 (无 ckpt 落盘) |
| `logs/task177/stage3_t5small_conformal_*.log` | Stage 3 训练日志 | 🗄️ 归档保留 |
| `dataset/Instruments/Instruments_t5_rqvae_posdep_conformal.npy` | Stage 2 SID | 🗄️ 归档 (broken baseline) |

## 5. 后续建议

1. **Phase 1 启动**: 立即进入 Task #178 (fixed baseline), 验证修复后 HG-Rec baseline 数字.
2. **Phase 4 可选**: fixed baseline 稳定后, 重跑 #177 (β conformal 位置依赖), 这时 λ_κ 才有真正的双曲意义.
3. **conformal vs sigmoid 选择**: 两个 Task (#176 sigmoid, #177 conformal) 都在 broken baseline 上终止, 没分出胜负. Phase 4 重新跑时要明确"哪个设计在 fixed baseline 上真有信号".

---

**Summary**: Task #177 在 broken baseline 9 epoch 处终止. Phase 0 已完成量化器代码-论文对齐 (6 处不匹配全部修复, 单元测试 4/4 通过). GPU 释放后 Phase 1 Task #178 立即启动.

result: Task #177 — broken baseline termination (Phase 0 修复后决策)