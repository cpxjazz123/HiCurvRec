---
task: 176
type: verdict
created: 2026-08-02
tags:
  - baseline
up: "[[index]]"
---
# Task #176 β(x) = sigmoid MCKG-style 位置依赖 — Broken Baseline 终止

> **任务目的**: 在 Stage 1 RQ-VAE 的 commitment loss 上叠加位置依赖 β(x) = β_base · sigmoid(α · pos_idx), 验证是否能提升 Musical_Instruments 上 Stage 3 的 R@10 (基线 HG-Rec #84 = 0.1020).

> **完成日期**: 2026-07-25
> **状态**: 🛑 Broken Baseline 终止 (Phase 0 修复后, 已在 GPU 1 上终止 PID 343349)

---

## 1. 执行时间线

| 阶段 | 时间 | 状态 |
|------|------|------|
| Stage 1 RQ-VAE 训练 (β sigmoid 位置依赖) | 2026-07-25 13:37 | ✅ 完成 |
| Stage 2 SID codebook inference | 2026-07-25 13:44 | ✅ 完成, `_t5_rqvae_posdep_sigmoid.npy` 落盘 |
| Stage 3 T5-small 5.5M 训练 (PID 343349) | 2026-07-25 14:12 起 | ⏸️ 进行到 epoch 9/200 (~30 min) 时被终止 |
| Stage 4 eval | - | ❌ 未启动 |

**Stage 3 训练进程** 在 broken baseline 上跑了约 30 min, 完成 9 epoch / 200 (early_stop=20 patience 未触发). 在 Phase 0.1-0.3 修复了 HG-Rec 量化器 6 处代码-论文不匹配之后, 决定终止并归类为 broken baseline 历史.

## 2. 关键指标

**Stage 1 + 2 产物** (在 broken baseline 上产出):
- `dataset/Instruments/Instruments_t5_rqvae_posdep_sigmoid.npy`: shape `(9922, 4)` int, 跟其他 broken-baseline SID 一并归档
- 保留作为历史参考, 不进入未来 Stage 4 评估

## 3. 分析解读

### 3.1 Broken baseline 范围

Stage 1 RQ-VAE 的代码-论文不匹配 (用户 2026-07-25 确认, Phase 0 验证):

| # | 论文 | broken 代码 | 影响 |
|---|------|------------|------|
| 1 | β 挂在 commitment (van den Oord) | `commitment + β·codebook_loss` (反向) | encoder 拿 2× 梯度, codebook 拿 0.5× |
| 2 | loss 输入在双曲流形上 | 球外点代入 poincare_distance | artanh 饱和, 梯度近零 |
| 3 | Eq (7) log∘exp = identity | 凭空多做一次 logmap0 | 范数发散 |
| 4 | 死代码 + 注释块残留 | 3 行死代码 (x_exp / cb_exp logmap0) | 无功能影响但混淆维护 |
| 5 | 论文 Table 6 默认值 | `[32,64,256]` β=0.25 | 跟 Instruments 论文原配不一致 |
| 6 | get_codebook 跟 forward 不一致 | 用不同码字表示 | 下游评估拿到错位码字 |

**结论**: Task #176 的 Stage 1 训练 + Stage 2 推断产物建立在有倾向坍缩结构的 broken code 上. 即使训完 Stage 3, R@10 也不能跟未来 fixed-baseline 对比.

### 3.2 Kill 决策信号

- ✅ Phase 0.1-0.3 完成 (utils.py 4 处修复 + train_hrqvae.py 默认值 + py_compile + 4/4 单元测试通过)
- ✅ GPU 1 (实际 cuda:0 → cuda:1) 被占, Phase 1/2/3 启动被堵
- ✅ Task #176 Stage 3 已经训 30 min, epoch 9/200, 未见收敛信号 (broken code 倾向持续)
- ✅ 修复后重跑 #176 在 fixed baseline 上 (Phase 4 可选任务) 更有意义

## 4. 产物清单

| 路径 | 说明 | 状态 |
|------|------|------|
| `products/task176/_TRAINING_PID` | 训练 PID 文件 | 🗑️ 已清 |
| `products/task176/_STAGE4_TRIGGER.sh` | Stage 4 自动触发脚本 | 🗑️ 已清 |
| `products/task176/posdep_sigmoid/` | Stage 1 ckpt | 🗄️ 归档保留 (broken baseline 标签) |
| `products/task176/t5small_posdep_sigmoid/jul-25-2026_14-12-03/Instruments/Jul-25-2026_14-12-55/` | Stage 3 T5 ckpt | 🗄️ 空目录 (无 ckpt 落盘) |
| `logs/task176/stage3_t5small_sigmoid_*.log` | Stage 3 训练日志 | 🗄️ 归档保留 |
| `dataset/Instruments/Instruments_t5_rqvae_posdep_sigmoid.npy` | Stage 2 SID | 🗄️ 归档 (broken baseline) |

## 5. 后续建议

1. **Phase 1 启动**: 立即进入 Task #178 (fixed baseline), 用 Task #84 verdict baseline = 0.1020 作交叉验证. 预期新 baseline 在 0.10-0.13 之间 (取决于修复影响几何信号 vs 数据过拟合特殊路径).
2. **Phase 4 可选**: 修正后基线稳定后, 重新跑 #176 (β sigmoid 位置依赖) 在 fixed baseline 上, 才能确认 sigmoid 位置依赖 vs baseline 是否真有差异. 不再做"在 broken 上训完" 的尝试.
3. **不要再用 broken 代码**: 未来任何 RQ-VAE 实验一律走 Phase 0 修复后的 utils.py + train_hrqvae.py 默认值.

---

**Summary**: Task #176 在 broken baseline 9 epoch 处终止. Phase 0 已完成量化器代码-论文对齐 (6 处不匹配全部修复, 单元测试 4/4 通过). GPU 释放后 Phase 1 Task #178 立即启动.

result: Task #176 — broken baseline termination (Phase 0 修复后决策)