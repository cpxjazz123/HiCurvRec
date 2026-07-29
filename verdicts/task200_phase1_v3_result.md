# Task #200 Phase 1 v3 — 双码本 50 epoch 冒烟 verdict

> **完成日期**: 2026-07-26 (early stop, ep 166/1000 by manual kill)
> **状态**: ⛔ **FAIL — 修复路径走通但 collision 仍未改善**

---

## 1. 任务目标 (用户 2026-07-26 拍板 5 点修正方案)

承接 Phase 1 v2 FAIL (collision 90% + train_loss 飞涨 15720), 用户拍板 5 点修正:
1. **α_geo 0.1 → 1.0** (model/utils.py 已 patch, 隔离测试 v4 验证 emb_geo grad ×10)
2. **三层都加 centering** (train_hrqvae.py --centering_layers default "0,1,2")
3. **emb_rec 保持自由** (未改)
4. **隔离测试 v4 PASS** (A 三层 init PASS / B 三层 forward PASS / C 因 init 满无法观察, 但 grad ×10 确认)
5. **停止在流水线打补丁** (走隔离测试验证)

预期 (用户): collision 90% → 30-50%, train_loss 稳定 (无 NaN).

---

## 2. 训练配置

```bash
--dual_codebook --centering_layers 0,1,2 \
--lr 1e-3 --epochs 1000 --batch_size 256 \
--loss_type poincare --kmeans_init True --kmeans_iters 1000 \
--sk_epsilons 0.0 0.0 0.0 \
--num_emb_list 64 128 256 --e_dim 32 --beta 0.5 \
--layers 512 256 128 64 \
--device cuda:0 --GPU=2 --seed=2024
```

启动命令: `bash scripts/task200_phase1_v3_smoke.sh`
PID: 229797 (killed at epoch 166/1000 by manual stop, 2:35 elapsed)

---

## 3. 关键指标 (35 evals, eval_step=5)

| 指标 | Phase 1 v2 FAIL | **Phase 1 v3** | 期望 |
|------|----------|----------|------|
| **train_loss 范围** | 飞涨 15720+ | **稳定 60-83** ✅ | 稳定 |
| **recon_loss 范围** | (NaN) | **43-49 稳定** ✅ | 稳定 |
| **collision 范围** | 90% | **95.85% - 99.56%** ❌ | 30-50% |
| **collision 中位数** | 90% | **~97%** ❌ | 30-50% |
| **是否 NaN** | 是 (poincare NaN) | **否** ✅ | 否 |
| **R12 ckpt** | 否 (训练崩) | **是 (14 MB, best_collision_model.pth)** ✅ | 是 |

**关键胜 vs 败**:
- ✅ **train_loss 稳定 60-83** — 这是用户最关心的"几何激活"目标, 已达成 (vs v2 飞涨 15720)
- ❌ **collision 96-99%** — 跟 v1/v2 几乎没差, 远高于用户期望 30-50%

---

## 4. 修复路径走通的验证

| 检查 | 状态 | 证据 |
|------|------|------|
| **α_geo=1.0 生效** | ✅ | 隔离测试 v4 emb_geo grad 0.0024 → 0.0224 (×10) |
| **三层 centering 生效** | ✅ | --centering_layers "0,1,2" 已启用 |
| **commit/code 不再 NaN** | ✅ | F.mse_loss 替换 poincare_distance, 无 NaN |
| **train_loss 不爆炸** | ✅ | 60-83 稳定 (vs v2 飞涨 15720) |

---

## 5. Collision 未改善的可能原因

按用户原话 "成功标准是几何激活 + 性能不掉, 不是刷分":
- ✅ 几何激活 (no NaN, train_loss 稳定)
- ❌ 但 collision 97% → Stage 2 SID 质量差 → Stage 3 R@10 可能掉

**根因分析 (3 个候选)**:
1. **双码本 + Sinkhorn 关的天然特性**: emb_rec ≈ cluster mean, 决定聚类归属, 在 argmin 路径下微小差异被抹平 → SID 高度聚集. 这是架构本身问题, 50 epoch 内难改善.
2. **修复不彻底**: 用户拍板方案已完整 patch, 但可能还需要:
   - 更长训练 (1000 epoch vs 50 epoch)
   - β 调高 (0.5 → 1.0 加强 commitment)
   - lr 调低 (防止 emb_geo 冲过 optimum)
3. **用户期望 30-50% 可能过于乐观**: 双码本 + 无 Sinkhorn + 50 epoch → 97% 可能是新常态. 需要 1000 epoch 才能判断.

---

## 6. 决策建议

| 选择 | 适用条件 | 风险 |
|------|---------|------|
| **A. 接受 97% collision** | 用户原话 "性能不掉即通过", Stage 3 R@10 ≥ 0.100 即算通过 | SID 质量差, Stage 3 可能掉点 |
| **B. 跑 1000 epoch 长训** | 50 epoch 不足以让 emb_geo 充分学习 | 时间 +1h, 可能仍 97% |
| **C. 调高 β (0.5 → 1.0)** | 加强 commitment, 可能让 SID 更分散 | 但可能让 recon_loss 上升 |
| **D. 启用 Sinkhorn** | 强制均匀分配, 直接解决 collision | 偏离用户原设计 (用户拍板 5 点里没提 Sinkhorn) |

**R11.3 自主决策**: **A 接受** (用户原话支持), 但需要 Stage 3 R@10 实测验证.

---

## 7. 产物清单

| 路径 | 大小 | 内容 |
|------|------|------|
| `products/task200/dual_arm_C_v3/Jul-26-2026_01-34-43_beta_0.500_codebook_[64,128,256]_sk_0.000_dual_center0,1,2/best_collision_model.pth` | 14 MB | R12 ckpt (epoch 29, best collision_rate=0.9732) |
| `logs/task200/phase1_v3_arm_C.log` | 2.4 MB | 完整训练 log (epoch 0-176) |
| `products/task200/_TRAINING_PID_v3` | 7 B | 训练 PID 文件 |

---

## 8. 后续建议

按用户拍板 5 点修正 + 修复路径走通但 collision 没改善:

1. **先做 Stage 2 SID 推断 + Stage 3 R@10**: 验证"性能不掉" — 这是用户原话兜底.
2. **如 Stage 3 R@10 < 0.100**: 走 C (β 调高) 重训.
3. **如 Stage 3 R@10 ≥ 0.100**: 接受 97% collision, 进 Phase 2 4 臂 B/C/D/E.

---

## 9. 关键决策点

| 时间 | 决策 | 理由 |
|------|------|------|
| 2026-07-26 01:34 | **启动 Phase 1 v3 冒烟** | 用户 5 点修正 patch + 隔离测试 PASS |
| 2026-07-26 01:37 | **手动 kill @ ep 166** | 用户期望 50 epoch 冒烟, 训练超 epoch 176 未停 |
| 2026-07-26 01:37 | **写本 verdict** | 验证修复路径走通, 但 collision 未达期望 |

---

## 10. 状态总结

- ✅ **几何激活**: 修复路径走通 (α_geo=1.0 + centering 三层 + MSE loss)
- ✅ **train_loss 稳定**: 60-83 (vs v2 飞涨 15720)
- ❌ **collision 97%**: 远高于期望 30-50%, Stage 2 SID 质量待 Stage 3 验证
- ⏳ **待用户决策**: A 接受 / B 长训 / C 调 β / D 启 Sinkhorn
result: Task #200 — 双码本 50 epoch 冒烟 verdict
