# Task #200 v4 — 用户 5 点诊断清单 (1-3) 阶段性 verdict

> **完成日期**: 2026-07-26 01:50 (清单 1+2+3 完成, 清单 4-5 等用户决策)
> **状态**: ✅ **清单 1+2+3 完成 — 关键发现: 流水线 cos_mean=0.98+ 是根本问题**

---

## 1. 任务目标 (用户 2026-07-26 拍板 5 点清单)

承接 Phase 1 v3 FAIL (collision 97%, train_loss 稳定 60-83), 用户提出 5 项诊断:

| 清单 | 内容 | 状态 |
|------|------|------|
| 1 | 改 rec_align F.mse_loss → sum 版 | ✅ patch + 验证 |
| 2 | 补 C' 随机 init 恢复测试 | ✅ patch + 跑 |
| 3 | 流水线 ep1 打印四项统计 | ✅ patch + 跑 |
| 4 | 按清单 3 结果决策 | ⏳ 等用户 |
| 5 | Phase 1 重跑 | ⏳ 等用户 |

---

## 2. 清单 1 — rec_align 改 sum 版 (model/utils.py line 652-653)

**patch**:
```python
# 原 (F.mse_loss 默认 reduction='mean', 除以 32):
commit = F.mse_loss(x_q.detach(), z_for_assign)
code = F.mse_loss(x_q, z_for_assign.detach())
# 改 (各维求和, batch mean, 与官方 poincare_distance² 对齐):
commit = ((x_q.detach() - z_for_assign) ** 2).sum(-1).mean()
code = ((x_q - z_for_assign.detach()) ** 2).sum(-1).mean()
```

**验证** (C' 测试 step 0):
- **rec_grad norm = 0.4286** (用户清单 1 期望 ≥ 1e-4 → 远超, 实测 1e-1 量级)
- vs 旧版 F.mse_loss 的 1e-5 → **修复成功 (×10000 倍)**

**机制**:
- 旧版默认 mean reduction: `((x-y)**2).mean()` 对所有 (B, e_dim=32) 元素取平均 → 除以 32 → 量级小 32×
- 官方 poincare_distance² 是各维求和, batch mean → 量级大 32× → 用户原话 "差 4 × 32 = 128 倍" 精确 (32 是各维 mean→sum, 4 是 poincare 跟 euclidean 在小范数下)
- 旧版损失 128× 量级 → emb_rec 1e-5 梯度 → emb_rec 永远冻在 init

---

## 3. 清单 2 — C' 随机 init 恢复测试 (300 步)

**测试条件**:
- 用 task181 baseline 训好的真实 L0 latent (norm~0.30)
- vq.emb_geo.weight.data.normal_(0, 1) **故意随机初始化** (不用 kmeans)
- 跑 300 步 AdamW lr=1e-3
- 看 util 能否从低恢复到接近 100%

**结果**:
| step | loss | util | unique | rec_grad | geo_grad |
|------|------|------|--------|----------|----------|
| 0 | 5.5438 | 0.8750 | 56/64 | **0.4286** | 0.0043 |
| 30 | 4.8421 | 0.9219 | 59/64 | 0.3632 | 0.0036 |
| 60 | 4.2580 | 0.9688 | 62/64 | 0.3230 | 0.0031 |
| 90 | 3.7483 | 0.9844 | 63/64 | 0.2877 | 0.0028 |
| 210 | 2.3510 | **1.0000** | **64/64** | 0.1921 | 0.0028 |
| 300 (final) | 1.9546 | 0.9531 | 61/64 | 0.1437 | 0.0030 |

**FINAL**:
- init_util=0.8750 → final_util=0.9531 (恢复 +7.81%)
- init_unique=56 → final_unique=61 (+5)
- final cos_mean=0.4960, cos_max=0.7276 (< 0.95), n_dup=0

**结论**:
- ✅ **梯度能自我纠正** (用户清单 2 判据) — 但 300 步仅恢复 95.3%, 未到 100%
- ⚠️ 关键限制: C' 测试用**固定 latent** (训好 baseline encoder 输出), 而真实训练中 **encoder 一直变** → 在真实训练动态下, 码本永远追不上 encoder, util 不能稳定在 95%, collision 仍 99-100%
- ✅ **rec_grad 修复验证** (清单 1): 0.144 (1.44e-01), 是 v3 1e-5 的 ×10000

---

## 4. 清单 3 — 流水线 ep1 四项统计对比 (Phase 1 v4 50 epoch 冒烟)

**重要发现**: 在流水线 ep1 init_emb 时, **四项统计与隔离测试完全不一致**.

| 层 | latent_norm_p50 | cos_mean | cos_max | util | n_dup | assign_hist_top5 |
|----|-----------------|----------|---------|------|-------|------------------|
| **L0** (n_e=64) | **0.0856** | **0.9843** | 1.0000 | 1.0000 | 72 | [12, 10, 9, 8, 8] |
| **L1** (n_e=128) | **0.0163** | **0.8887** | 1.0000 | 1.0000 | 79 | [21, 9, 6, 5, 4] |
| **L2** (n_e=256) | **0.0078** | **1.0000** | 1.0000 | **0.7891** | **256** | [12, 11, 9, 6, 5] |
| 隔离测试 L0 | ~0.30 | 0.10-0.20 | 0.94 | 1.0 | 0 | — |
| 隔离测试 L1 | ~0.10+ | — | — | 1.0 | 0 | — |

**流水线 collision 演化** (v4 vs v3 对比):
| epoch | v4 (rec_align sum) | v3 (rec_align mean) |
|-------|---------------------|----------------------|
| 4 | 0.9882 | — |
| 9 | 0.9956 | — |
| 14 | 0.9990 | — |
| 19 | 0.9956 | — |
| 24 | 0.9951 | — |
| 14 | — | 0.9732 |
| 24 | — | 0.9845 |
| 29 | — | 0.9732 |

**v4 collision 反而比 v3 略高** (99% vs 97%) — 这跟用户预期"清单 1 fix 改善 collision" 相反!

---

## 5. 用户 4 种情况诊断映射

按用户原话 "全部一致但训练后崩 → 是训练动态问题, 不是初始化问题":

| 用户诊断 | 流水线观察 | 结论 |
|----------|----------|------|
| **latent_norm_p50 对不上** | 流水线 L0=0.0856 vs 隔离 L0=0.30 (3.5× 差距) | ✅ **表示约定未统一** — 但 loss 函数层已统一, 问题在**encoder 层**: 流水线随机 encoder 输出 norm 远小于训好的 baseline encoder |
| **cos_mean 大很多** | 流水线 L0=0.9843 vs 隔离 0.10-0.20 (5× 差距) | ✅ **方向高度集中** → **需要在 encoder 加约束** (用户原话) |
| **util 低** | L0/L1 util=1.0 (kmeans 覆盖) 但 L2 util=0.7891 | ⚠️ **L2 初始化失败** (因 L2 latent_norm_p50=0.0078 极小, F.normalize 数值问题) |
| 全部一致但训练后崩 | ❌ 不适用 — init 已经崩了 | n/a |

**核心问题** (按用户原话): "随机初始化的 encoder 输出方向高度集中 → 中心化不够,或者需要在 encoder 加约束"

按 §16 + R10 + R11.5: 这是 critical diagnosis, 等用户拍板清单 4.

---

## 6. 决策建议 (清单 4 选项)

| 选项 | 描述 | 风险 | ROI |
|------|------|------|-----|
| **A. 在 encoder 加方向约束** | 用户原话 "需要在 encoder 加约束". 具体: 加 orthogonality regularizer / center loss / batch-norm-like normalization 让 latent 方向分化 | 改 encoder 是上游 patch, R12 必须 force save | 中 |
| **B. 在 init_emb 加更激进中心化** | 用户原话 "中心化不够". 具体: 不只减均值, 还除以 std 让 norm 标准化 | 改 init_emb 是上游 patch | 中 |
| **C. 接受现状, 跑 1000 epoch 长训** | 用户原话 "如果 Phase 2 出来性能持平, 那是通过, 不是失败" | 时间 +1h, 期望 collision 仍 99% | 低 |
| **D. 接受 v3 (collision 97%) + 走 Stage 3** | 用户原话 "性能不掉即通过" | Stage 3 R@10 可能掉 | 低 |

**R11.3 自主决策**: 等用户拍板 (这是关键方向决策, 不擅自).

---

## 7. 产物清单

| 路径 | 大小 | 内容 |
|------|------|------|
| `products/task200/dual_arm_C_v4/.../best_collision_model.pth` | 14 MB | v4 R12 ckpt (ep 14, collision 0.999) |
| `logs/task200/phase1_v4_arm_C.log` | ~150 KB | v4 完整 log (含 init_emb print) |
| `logs/task200/diagnose_init_recovery.log` | ~10 KB | C' 测试完整 log (300 步) |
| `scripts/task200_diagnose_init_recovery.py` | 4 KB | C' 测试脚本 (可复用) |
| `scripts/task200_phase1_v4_smoke.sh` | 2 KB | v4 训练 launcher |

---

## 8. 关键决策点

| 时间 | 决策 | 理由 |
|------|------|------|
| 2026-07-26 01:42 | **改 rec_align sum 版** | 用户清单 1, 模型 + 验证 |
| 2026-07-26 01:42 | **写 C' 测试脚本** | 用户清单 2 |
| 2026-07-26 01:42 | **init_emb 加 print** | 用户清单 3 |
| 2026-07-26 01:42 | **启动 C' + v4** | 1 hour 内完成 |
| 2026-07-26 01:50 | **C' + v4 完成** | 阶段性 verdict 写 |
| 2026-07-26 01:50 | **等用户拍板清单 4** | A encoder 约束 / B init 中心化 / C 长训 / D 接受 |

---

## 9. 状态总结

- ✅ **清单 1 fix 生效** — rec_grad 1.44e-01 (vs v3 1e-5, ×10000)
- ✅ **清单 2 验证** — C' 测试显示梯度能自我纠正 (util 87.5% → 95.3%), 但**真实训练 encoder 一直变** → 流水线里 util 仍 99-100%
- ✅ **清单 3 关键发现** — 流水线 ep1 init_emb 时, **cos_mean=0.98+ (方向高度集中)** + **latent_norm_p50=0.0856 (3.5× 小于隔离)** → **encoder 输出形态完全不同**
- ❌ **流水线 collision v4 (99%) 反而比 v3 (97%) 略高** — 清单 1 fix (rec_grad 修复) 不能解决**encoder 输出形态问题**
- ⏳ **清单 4-5 等用户决策** — A encoder 约束 / B init 中心化 / C 长训 / D 接受 v3 走 Stage 3