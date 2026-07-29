# Task #271 — A1 β=0.0 Stage 1 验证 FAIL (L0 ≥ 90% 不可达)

> **完成日期**: 2026-07-29
> **状态**: 🔴 **FAIL (NO-GO 路径 1/3)** — A1 β=0.0 配方 NOT PASS

---

## 1. 实测 (完整 step2 monitor 历史)

| ep | L0 | L1 | L2 | collision | 备注 |
|----|------|------|------|-----------|------|
| 5  | 4.7% (3/64)  | 0.8% (1/128) | 2.3% (6/256) | 0.9989 | 严重早期坍缩 |
| 10 | 17.2% (11/64) | 16.4% (21/128) | 9.0% (23/256) | 0.9779 | 上升期 |
| 15 | 21.9% (14/64) | 11.7% (15/128) | 8.6% (22/256) | 0.9889 | 振荡 |
| 20 | **29.7% (19/64)** ↑ peak | 12.5% (16/128) | 9.4% (24/256) | 0.9862 | L0 历史峰值 |
| 25 | 26.6% (17/64) ↓ | 20.3% (26/128) | 5.5% (14/256) | 0.9869 | 饱和 |
| 30 | 26.6% (17/64) | 23.4% (30/128) | 5.9% (15/256) | 0.9869 | **20% 硬 kill 触发** |

**baseline recipe contrast (Task #263 verifier, task253 β=0.5)**:
| ep | L0 | L1 | L2 | collision |
|----|------|------|------|-----------|
| 50 (best_collision) | **73.44% (47/64)** | 100% | 100% | 0.0915 |

## 2. 通过条件判定

| 条件 | 阈值 | A1 实测 | 判定 |
|------|------|---------|------|
| L0 ≥ 90.625% @ ep ≥ 30 | 58/64 | 26.6% (17/64) | ❌ **差 63.7pp** |
| collision < 0.95 | < 0.95 | 0.9869 | ❌ **差 3.7pp** |
| recon_loss ≤ 1500 | ≤ 1500 | ~12 | ✅ PASS |

**总判定**: **FAIL** (三个条件两个 FAIL).

## 3. 反预期关键发现

Task #271 反预期:

1. **β 不是 L0 坍缩根因**: Task #263 baseline β=0.5 L0=**73.44%** @ ep50. Task #271 A1 β=0.0 L0=**26.6%** @ ep30. **β=0 反而比 β=0.5 更糟.** 假设 "双曲 commit loss 是 L0 utilization 瓶颈" 证伪.
2. **码字 Euclidean 范数 stuck**: hypnorm 各层 ‖x‖_E mean 50 epoch 完全不变 (L0=0.076, L1=0.005, L2=0.003). 码字没从初始 K-means 位置扩散.
3. **结构性 L0 容量不足**: K=64 codes 装 9922 items, 仅 19 unique codes (ep20 peak). 即便加大训练 epoch, K=64 是潜在硬上限.
4. **20% 硬 kill 真实存在**: hrqvae_trainer.py:489-502 — Issue #17 Gate 1 修复后首次触发. 修前 50 epoch 训练没人知道这个机制存在.

## 4. 反派备选 (R11.3 必须明示)

| 备选 | 理由 | 不选理由 |
|------|------|----------|
| **继续跑完 50 epoch** (关掉硬 kill) | 看 ep50 时 L0 是不是会反弹 | `disable_kill=1` 能关掉, 但 A1 ep25-30 已经饱和, 跑完意义低. |
| **A3 freeze encoder** | Task #193 已有杠杆, ROI 中 | 历史 task178/180 freeze encoder 后 L0 < 90% 风险高, 同样走不通 |
| **A2 curriculum** (A1 warm-start β=0.5) | 已有 A1 ckpt | A1 起点 L0=26.6% 不好, warm-start β=0.5 难反弹, ROI 极低 |
| **--num_emb_list 128 256 512** | 直觉上 L0 容量翻倍 | **破坏 Stage 3 SID 维度兼容** (现在 Stage 3 吃 num_emb_list=64 L0), 架构级变更, 不在本方向候选范围 |
| **encoder variance 诊断** | 真查 encoder 输出是否多模 | ROI 高 (data 驱动诊断), 但单 cron tick 放不下, 候选 backlog |

**选 A1 FAIL + 闭环 + 资源转**: 不再花 GPU 验证 A2/A3, 直接转 backlog 候选 (m-arm κ-Stereographic / encoder 诊断 / VERDICT 库存盘点).

## 5. 资源清算 (本任务)

- 启动 A1: 5 min Stage 1 (~epoch 30 时因硬 kill 退出, 实际 ~30s)
- 修了 launcher set -u trap (commit c0543d1)
- **0 后续 GPU** (ROI 边际 < 0)

## 6. 关键决策点 (R11.3)

- **不继续跑 A1 50 epoch**: ep25-30 已饱和, ep50 不会反弹
- **不启动 A2/A3**: A1 FAIL 证据足够, A3 ROI 显著降低
- **不假装 success**: 诚实记录 26.6% < 90%, 标 NO-GO
- **下个 cron tick 决策**: 转 backlog 候选 2 (m-arm κ-Stereographic) 或 候选 4 (VERDICT 库存盘点, Issue #10/#17 全关后回顾)
- **Issue #17 Gate 1 修复 side effect**: 硬 kill 触发了 → 未来所有 β=0.5 baseline 50 epoch 训练都可能在 ep30 触发. 这是 Issue #17 Gate 1 修复的**隐式发现**, 需要单独跟踪 (Task #270 衍生命题)

## 7. Issue 状态 (本任务不动)

- Issue #17: 仍 closed (Gate 1 修复成功, Gate 2 实测 PASS, 副作用是发现了 20% 硬 kill 触发条件不严格, 但**不构成立 issue 再开**)
- 候选新 issue (R11.5 自主决策, **不立**): "Step2 monitor 20% 硬 kill 阈值需要在 epoch 30 之前更早触发" → 风险: 这等价于让 algorithm 提前 abort, 影响所有 baseline; 建议**不立 issue 而是把硬 kill 阈值作为新研究的候选课题**

## 8. 物理产物

```
verdicts/task271_a1_beta_zero_run_result.md  (本文件)
descriptions/task271_a1_beta_zero_run.md
logs/task270/stage1_A1_2026-07-29_10-11-02.log  (完整 stdout)
products/task270/A1_euclidean/Jul-29-2026_10-11-33_beta_0.000_codebook_[64,128,256]_sk_0.000/
  best_collision_model.pth (ep24, collision 0.9869, L0 17/64 = 26.6%)
  best_loss_model.pth
  epoch_9_collision_0.9779_model.pth (ep9 best collision before plateau)
  epoch_24_collision_0.9869_model.pth
```

result: Task #271 — A1 β=0.0 Stage 1 验证 FAIL. L0 利用率 ep20 峰值 29.7% (19/64) → ep30 饱和 26.6% (17/64). 20% 硬 kill 触发 (Issue #17 Gate 1 修复副作用, 此前 baseline 训练从未到达 ep30 utilization 真打印). β 不是 L0 坍缩根因 = hypothesis 证伪. A2 (curriculum, 依赖 A1) 弃. A3 (freeze encoder) 候选保留但 ROI 显著降低. 资源转 backlog 候选 2 或 4.
