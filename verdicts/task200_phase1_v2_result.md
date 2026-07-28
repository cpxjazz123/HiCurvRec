# Task #200 Phase 1 v2 — 双码本 forward 修复方向 D 重跑 (FAIL 2 阶段)

> **状态**: ⛔ Phase 1 v2 FAIL (修复方向 D 部分生效, 架构层仍有梯度爆炸问题)
> **完成日期**: 2026-07-26
> **下一阶段**: 等用户决策 (修复路径选项见 §5)

---

## 1. 实验概要

**用户拍板修复方向 D (2026-07-26)**: commit/code 用欧氏 MSE (而非 poincare_distance)。emb_geo / emb_rec / latent 全部在切空间,expmap0 只在算 d 和 geo_loss 两行出现一次。

**Patch 内容** (HG-Rec/model/utils.py `HVectorQuantization.forward`):
```python
# Before (v1, FAIL):
commit = torch.mean(poincare_distance(x_q.detach(), z_for_assign, self.c) ** 2)
code = torch.mean(poincare_distance(x_q, z_for_assign.detach(), self.c) ** 2)

# After (v2):
commit = F.mse_loss(x_q.detach(), z_for_assign)
code = F.mse_loss(x_q, z_for_assign.detach())
```

init_emb 不变(已正确)。

**启动参数** (臂 C, 50 epoch target / 实际跑到 epoch 388 后 kill, GPU 2):
```bash
python3 -u train_hrqvae.py \
    --data_path $DATA --lr 1e-3 --epochs 1000 --batch_size 256 \
    --loss_type poincare --kmeans_init True --kmeans_iters 1000 \
    --sk_epsilons 0.0 0.0 0.0 --sk_iters 50 \
    --num_emb_list 64 128 256 --e_dim 32 \
    --beta 0.5 --layers 512 256 128 64 \
    --dual_codebook --centering_layers 0 \
    --ckpt_dir $SAVE_DIR
```

---

## 2. 实验结果

| 阶段 | Epoch | train_loss | recon_loss | collision_rate | 状态 |
|------|-------|-----------|-----------|----------------|------|
| **Stage 1 (稳定期)** | 14-130 | **65-70** | 37-43 | 90-91% | ✅ 修复生效 (vs v1 卡 5800) |
| **Stage 2 (崩塌期)** | 380 | **12,552** | 46.5 | — | ❌ 开始飞涨 |
| | 384 | 14,064 | 46.6 | **98.7%** | ❌ |
| | 388 | **15,720** | 46.7 | — | ❌ 持续上升 |

---

## 3. 关键诊断

### 3.1 修复方向 D 部分生效(Stage 1)

- ✅ **train_loss 稳定 65-70**(Phase 1 v1 是 5800 卡死) — 修复方向 D 确认有效,commit/code 改 MSE 解决了 poincare_distance 在 ‖x‖>1 时 lambda_x 变负爆 NaN 的根本问题
- ✅ **recon_loss 37 稳定下降**(decoder 学习正常)
- ❌ **collision_rate 90%**(阈值 30% 仍未达) — argmin 不可微 → emb_geo 唯一梯度来自 α=0.1 geo_loss,梯度强度不够

### 3.2 Stage 2 train_loss 飞涨(架构层问题)

**症状**: train_loss 65 → 15720 (epoch 130 → 388), recon_loss 稳定 37-47。

**可能根因**:
1. **emb_rec 切空间任意尺度 + MSE 链式梯度不稳定**:emb_rec init 用 latent 均值(范数 ~5-10),MSE loss 推动 emb_rec,但 decoder 输入 emb_rec 任意尺度,decoder 输出任意尺度,decoder 反向推 emb_rec 可能进入不稳定区域
2. **z_mean EMA 偏移累积**:L0 centering z_mean 训练时 EMA 更新,长期训练 z_mean 可能偏离数据均值,z_for_assign 范数变化
3. **geo_loss 在 d 增大时累积**:α_geo=0.1 在 d 大时贡献 0.1 × mean(d),d 增大 → geo_loss 增大

---

## 4. 上游 PATCH 状态

| 文件 | 状态 | 备份 |
|------|------|------|
| HG-Rec/model/utils.py | patched (forward commit/code 改 MSE) | `.bak200` |
| HG-Rec/model/hrqvae.py | patched (双码本入口) | `.bak200` |
| HG-Rec/model/hrqvae_trainer.py | patched (ckpt 目录加 _dual 后缀) | `.bak200` |
| HG-Rec/train_hrqvae.py | patched (--dual_codebook + --centering_layers) | `.bak200` |

如需 rollback:
```bash
cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec && \
  \cp -f model/utils.py.bak200 model/utils.py && \
  \cp -f model/hrqvae.py.bak200 model/hrqvae.py && \
  \cp -f model/hrqvae_trainer.py.bak200 model/hrqvae_trainer.py && \
  \cp -f train_hrqvae.py.bak200 train_hrqvae.py
```

---

## 5. 进一步修复路径(R11.4 仅记录,等用户拍板)

| 选项 | 改动 | 预期效果 |
|------|------|---------|
| **E1** | emb_rec 加 F.normalize (限制范数=1) | 限制 emb_rec 任意尺度,避免 decoder 反推梯度进入不稳定区 |
| **E2** | 调小 α_geo (0.1 → 0.01) | 减少 geo_loss 对 emb_geo 的弱梯度 |
| **E3** | 增加 emb_rec gradient clipping (per-param ‖g‖ ≤ 1.0) | 限制 emb_rec 梯度爆炸 |
| **E4** | 关闭 z_mean EMA (改 fixed z_mean) | 排除 EMA 累积偏移影响 |
| **E5** | 缩短训练 (50 epoch 立刻停) | 第一阶段稳定期内取 ckpt,放弃长训 |
| **E6** | 放弃双码本,emb_geo 仅作 kmeans init 辅助(等价强化 kmeans init) | 等价判 Phase 2 NO-GO,不再调架构 |

**R11.4 决策点**: Stage 1 修复成功 → 方向 D 有效; Stage 2 崩塌 → emb_rec 切空间任意尺度需要额外约束(E1/E3),或者换架构(E6)。等用户拍板。

---

## 6. 产物清单

- `products/task200/dual_arm_C_v2/` — Phase 1 v2 训练 ckpt (已 kill, 保留 best_collision/best_loss)
- `logs/task200/phase1_v2_arm_C.log` — 完整日志 (388 epoch, 12.5 KB)
- `scripts/task200_phase1_v2_smoke.sh` — v2 launcher (50 epoch target)
- `scripts/task200_phase0_v4_diagnose.py` — Phase 0 v4 诊断脚本 (np import bug 待修)
- `verdicts/task200_phase0_result.md` — Phase 0 PASS
- `verdicts/task200_phase1_result.md` — Phase 1 v1 FAIL
- `verdicts/task200_phase1_v2_result.md` — Phase 1 v2 (本文档)

---

## 7. 后续建议

**当前状态**: Phase 0 PASS / Phase 1 v1 FAIL (commit/code 数值爆) / Phase 1 v2 部分通过 (Stage 1 修复确认 + Stage 2 架构崩塌)。

**§16 backlog**:
- #201 Stage 3 逐层 κ — 等 #200 通过
- #196 γ 扫描 / #197 双码本 Stage 2 / #198 逐层 κ — §16 待启动队列

**下一步 (R11 自主决策)**:
- 方向 D 已确认是修复 commit/code 数值的正确方向, 但 emb_rec 切空间任意尺度 + MSE 链式梯度有架构层问题
- 建议用户拍板: E1 (emb_rec 归一化) + E3 (grad clip) 组合, 重新跑 Phase 1 v3
- 也可接受 E6 (放弃双码本) 直接关停 #200
- GPU 0/2/3 空闲, 等决策

result: Phase 1 v2 部分通过 — 修复方向 D 解决了 v1 commit/code 数值爆炸 (train_loss 5800 → 65-70), 但 Stage 2 (epoch 200-388) train_loss 飞涨 70 → 15720, collision 98.7%, 根因是 emb_rec 切空间任意尺度 + MSE 链式梯度在长训不稳定。等用户拍板进一步修复 (E1-E6)。