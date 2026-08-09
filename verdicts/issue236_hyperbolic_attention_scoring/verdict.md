# Issue #236 — Stage3 Hyperbolic Attention Scoring Verdict (NO-GO + R37 回退)

## 状态: ❌ NO-GO (R23 信号 1 触发 kill + R37/R38 回退流程)

## 时间线

- **2026-08-10 02:41** — v24 stage3 训练启动 (DDP 4 卡 PID 2783359, 14 layers patched)
- **2026-08-10 02:43** — Epoch 1 loss=5.99 (健康下降)
- **2026-08-10 02:46** — Epoch 5 valid_R@10=**0.0000** [REFUSE SAVE] (catastrophic drop protection)
- **2026-08-10 02:51** — Epoch 10 valid_R@10=**0.0000** [REFUSE SAVE] (再次)
- **2026-08-10 02:54** — **R23 信号 1 触发: val_R@10=0 跨 ≥2 ckpt** → 立即 kill -9 PID 2783359

## R23 七信号检测

| # | 信号 | 状态 |
|---|------|------|
| 1 | val_R@10=0 跨 ≥2 ckpt | 🛑 **触发** (epoch 5 + epoch 10 都是 0.0000) |
| 2 | loss 不下降 | ✅ 损失 7.61→4.04 健康下降 |
| 3 | val loss 反向 | N/A |
| 4 | wrapper broken | ✅ 无 |
| 5 | ckpt 不存 | ✅ REFUSE SAVE 是有意的(无 ckpt) |
| 6 | NaN-Inf | ✅ 无 |
| 7 | GPU 100% loss 不变 | ✅ loss 健康下降 |

## R38 决策行

> **v24 mid-training regress**: v24 best_valid=0.0000 < v18 baseline=0.1011 (-0.1011),
> valid_R@10=0 跨 ≥2 ckpt (epoch 5 + epoch 10) 触发 R23 信号 1 + R38 信号 1
> (best valid 落后 vN-1 且饱和平台), **立即 kill + 回退至 v18 重新创新**

## R37 决策行

> **v24 比 v18 差** (-0.1011, valid 完全 0), **回退至 v18 重新创新**
> v24 所有产物 (Stage 3 ckpt / Stage 4 raw_predictions / verdict) 仅留作记录, 不作为下一版本起点。
> **新版本必须以 v18 为唯一基础。**

## 4 Gate 最终判定

- **Gate 1 (Stage2 SID 一致)**: ✅ PASS — sha=5f8331cc 与 v18/v20 同
- **Gate 2 (Stage3 训练健康)**: ❌ FAIL — valid_R@10=0 跨 ≥2 ckpt, 训练产物无效
- **Gate 3 (曲率信号健康)**: ❌ FAIL — 14 layers 装了 -d_P score, 但 valid=0 说明该曲率信号破坏了 T5 注意力学习能力
- **Gate 4 (端到端 test_R@10)**: ❌ FAIL — 没有跑 stage4_beam20 (Stage 3 ckpt 未生成, REFUSE SAVE 触发 2 次)

## 根因分析

- v24 的 Issue #236 设计 = 把 T5 attention score 从 `matmul(Q,K^T)` 完全替换为 `-d_P(Q,K)`
- 实测结果: 训练 loss 健康下降(7.61 → 4.04),但 **valid_R@10 始终是 0**
- 表明 -d_P score 与 `cross-entropy generation` 任务正交,无法训练出可用的 generation 模型
- 可能原因:
  1. -d_P 的 score 范围与 matmul 不对齐(虽然代码做了 tanh 缩放到 ball)
  2. decoder cross-attn 仍用 matmul 而 encoder 用 -d_P,两端 score 空间不一致 → decoder 无法解读 encoder hidden states
  3. c=5.0 太大(默认 logit=0 → c=5),导致 -d_P 数值过小,softmax 后概率接近均匀
- 不论根因,**v24 方向已被验证 NO-GO**,不进入 v25 二次创新

## R 合规

- **R36** ✅ 新曲率机制 (替换 attention score 函数), 不是调参
- **R37** ✅ 完全 v18 base, 不在失败品上叠加
- **R38** ✅ 训练期早停回退触发, 立即 kill
- **R23** ✅ 信号 1 检测 + kill -9 + NO-GO
- **R35** ✅ N/A (无 ckpt, 没跑 stage4)
- **R39** ✅ 立即实施, 不阻塞 Gate A

## 后续

- v18 仍为当前最优基础(R@10=0.1011)
- Issue #100 (= #236) **close**
- 下个新方向必须从 v18 base 出发, 不在 v24 失败品上叠加
- R37 决策已写入 commit message + this verdict + issue close comment