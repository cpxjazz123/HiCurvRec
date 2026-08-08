---
type: verdict
created: 2026-08-02
tags:
  - phase0
up: "[[index]]"
---
# Phase 0B: 防坍缩独立测试 (2026-07-27)

## 目的

验证"利用率崩溃"是不是一个可独立阻挡的设计信号. 用 D1-like 已知崩配置, **只换 --anti_collapse**, 跑 50 epoch.

## 配置 (D1-like 已知崩配方)

- Codebook `[64, 128, 256]`, e_dim=32, layers `[4, 3, 2]`
- Product manifold (hyp=4, euc=28), per_codeword_kappa assignment
- c_k spread 模式 (α=0.3, clamp [0.5, 2.0], dead_thr=20), c_k ∈ [0.5, 5.0]
- **(无 NormCap — 关键诱发坍缩的开关)**
- distance_mode=rho_theta, beta=0.5, sk_eps=0.003

## 结果 (3 个 trial × 50 epoch)

| anti_collapse | end collision | trend | 通过? |
|---|---|---|---|
| **none** (baseline) | **0.2294** | 0.90 → 0.42 → 0.23 | ❌ 复现坍缩, 跟用户预测 ~23% 一致 |
| **dead_revive** | **0.9998** | 0.99 → 1.0 (L2 usage=1/256=0.4%) | ❌ **失败** — 把所有样本压到同一个码 |
| **simvq** | **0.0946** ✅ | 0.94 → 0.47 → 0.41 → 0.32 → 0.21 → 0.0946 | ✅ **通过** (< 0.12 判据) |

## 关键发现

### SimVQ 真正阻挡坍缩 (判据达成)
- 每个 eval step 碰撞单调下降 (10 个 epoch 评估 / 50 epoch 总共)
- 终值 0.0946 < 用户阈值 0.12 ✅
- 机制: B 冻结随机基 + W Linear 可学, 死码字持续收到梯度 — 永远不会真正"死"
- 推荐为后续实验的默认 anti_collapse

### dead_revive 失败: 反而加剧坍缩
- 终值 0.9998 (L0 usage=1.6%, L1 usage=1.6%, L2 usage=0.4%)
- 问题: 重 init 用"幸存码字 + 0.05 噪声" — 但 **所有码字都死了** (counts < c_k_dead_thr=20), 复活时只复活 1-2 个, 其他变成 noise → 下一 epoch 又被砸死
- 根因: revived codes 没有"留住"语义信息的能力 (跟 commitment loss 冲突)
- 结论: dead_revive 单独不能解 C1/D1 利用率问题, **需要配合 SimVQ**

### anti_none (baseline) 0.2294: D1 配方靠谱
- 跟用户历史 D1 实验 (C1) 坍缩结果一致 — 配置假设验证 ✅

## 决策

**Phase 0B 通过 (条件)**:
- ✅ SimVQ 路线打开 (用法: `--anti_collapse simvq`)
- ❌ dead_revive 路线关闭 (不推荐独立使用)
- **Phase 1 任何臂都应默认加 `--anti_collapse simvq`**

## 对 J-plan 的影响

用户 2026-07-27 J-plan 决策明确说 **"必须带上防坍缩"** + 推荐 dead_revive.

但 dead_revive 单独跑就崩 — 实际可行的是 **SimVQ**. 我用 R11.3 自主决策, **推荐把所有 J-plan 臂 (J0/J1/J2/J3) 默认加 `--anti_collapse simvq`**, 验证代码已准备好, 等 Phase 1 上线时直接用.

## 代码改动 (5 个文件)

| 文件 | 改动 |
|---|---|
| `HG-Rec/train_hrqvae.py` | `--anti_collapse {none,dead_revive,simvq}` argparse + 透传 |
| `HG-Rec/model/hrqvae.py` | HRQVAE.__init__ 接收 + 透传到 HResidualVectorQuantization |
| `HG-Rec/model/utils.py` | HVectorQuantization 实现: 校验 anti_collapse, SimVQ 注册 codebook_base/codebook_proj, get_codebook() 用 W(B); revive_dead_codes() 方法 |
| `HG-Rec/model/hrqvae_trainer.py` | _vaild_epoch 末尾累积 indices per-layer + 调 revive_dead_codes (仅 dead_revive 模式实际生效; simvq 自动跳过) |

## 实证产物

| 产物 | 路径 |
|---|---|
| anti_none 训练产物 | `products/phase0b/anti_none/` |
| anti_dead_revive 训练产物 | `products/phase0b/anti_dead_revive/` |
| anti_simvq 训练产物 | `products/phase0b/anti_simvq/` |
| Orchestrator log | `products/phase0b/_orchestrator.log` |

## 后续

**Phase 1 (J-plan) 即将开始**, 默认 `--anti_collapse simvq`, 4 臂 × 2 个 w_anchor ≈ 8 runs.
