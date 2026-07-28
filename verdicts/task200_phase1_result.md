# Task #200 Phase 1 — 双码本冒烟测试 FAIL verdict

> **状态**: ⛔ Phase 1 FAIL (设计层 bug, 不可继续 patch)
> **完成日期**: 2026-07-26
> **下一阶段**: 等用户决策 (Phase 2 设计需重新审视)

---

## 1. 实验概要

**任务**: 用户 2026-07-26 设计的双码本几何解耦方案 Phase 1 冒烟测试
- dual_codebook = (emb_geo 球面 kmeans 方向 + emb_rec 残差均值) 双码本
- L0 centering (z_mean EMA) 防止锥体坍缩
- 臂 C: K=[64,128,256], ρ 目标 2.0/2.7/3.4

**启动参数** (50 epoch smoke, GPU 2):
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

## 2. 实验结果(50 epoch 实际跑到 epoch 46 后 kill)

| 指标 | 用户阈值 | 实际值(epoch 44) | 状态 |
|------|---------|----------------|------|
| NaN | 无 | 无 | ✅ |
| cos_max 稳定 | 不上升 | **无法测**(码本未更新) | ❌ |
| **collision_rate** | **< 30%** | **96.41%** | **❌ 严重失败** |
| train_loss | 稳定下降 | 5853.14→5852.65(epoch 40-46 卡死) | ❌ |
| **recon_loss** | 正常 | **39.93**(正常) | ✅ |
| **‖x‖_E (latent 范数)** | > 0 | **0.000 (全部 3 层)** | **❌ 设计层 bug** |

---

## 3. 根因诊断(关键)

**核心问题**:dual_codebook=True 路径下,latent 在 HRQVAE.forward 已被 `logmap0` 投影到切空间(欧几里得),但 HVectorQuantization.init_emb 仍然执行:

```python
# utils.py dual_codebook init_emb:
latent = proj_to_ball(expmap0(latent, c=1), c=1)
```

切空间向量被强制 `expmap0` 推到 Poincaré ball,然后 `proj_to_ball` 由于范数 ≫ 1 全部截断到 origin 邻域 → **`latent ≈ 0`**。

**后果链**:
1. `‖latent‖_E ≈ 0` → 三层 norm 全部 0.000(与 hypnorm log 一致)
2. `dirs = F.normalize(latent)` → 数值 NaN 风险,或方向随机
3. emb_geo 球面 kmeans 初始化在**接近全 0 的 latent 上**,中心点坍缩到一处
4. forward 中所有 z 都量化到同一个码字 → collision 96%
5. **geo_loss 项也炸到 5812**(`(soft_assign * d).sum()` 中 d 不收敛)

**Phase 0 v3 PASS 的原因**:Phase 0 是 standalone 验证,从 `data = torch.load(...)` 直接加载 latent,**不走 HRQVAE.encoder + logmap0**,所以 latent 是 Poincaré ball 上的真实点(范数 ~0.85),proj_to_ball 正常。

**Phase 1 失败的本质**:`init_emb` 假设输入是 Poincaré ball 点,但实际接收到的是 logmap0 后的切空间向量。

---

## 4. 上游 PATCH 状态(若需 rollback)

| 文件 | 状态 | 备份 |
|------|------|------|
| HG-Rec/model/utils.py | patched | `.bak200` |
| HG-Rec/model/hrqvae.py | patched | `.bak200` |
| HG-Rec/model/hrqvae_trainer.py | patched | `.bak200` |
| HG-Rec/train_hrqvae.py | patched | `.bak200` |

**当前双码本代码保留在磁盘**(用户可决定保留或 rollback)。代码改动包括:
- `--dual_codebook` / `--centering_layers` argparse
- HVectorQuantization.dual_codebook forward path(带 geo_loss)
- HRQVAE.use_centering_list 透传
- 目录 `_dual_center<L>` 后缀

如需 rollback:`cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec && \cp -f model/utils.py.bak200 model/utils.py && \cp -f model/hrqvae.py.bak200 model/hrqvae.py && \cp -f model/hrqvae_trainer.py.bak200 model/hrqvae_trainer.py && \cp -f train_hrqvae.py.bak200 train_hrqvae.py`

---

## 5. 修复路径(R11.4 仅作记录,不主动 patch)

**修复方向 A(最小改动)**: 在 dual_codebook init_emb / forward 中,**先 expmap0** 把切空间 latent 拉回 Poincaré ball,然后**末尾 logmap0** 还原成切空间做后续计算。也就是在 HRQVAE → HResidualVectorQuantization → HVectorQuantization 边界显式做一次转换。

**修复方向 B(改设计)**: 取消 dual_codebook 中的 proj_to_ball(expmap0(latent)),**让 emb_geo/emb_rec 直接处理欧几里得向量**(切空间),几何激活改成"在 expmap0 之后做"。这等于完全重构 forward,需要重新做 Phase 0。

**修复方向 C(回退双码本)**: 改回标准 VQ-VAE argmin + emb_geo 仅作初始化辅助,放弃 forward 路径,实质等价于 kmeans init 的强化版。

**R11.4 决策点(等用户)**:
- 选 A: 最小改动重跑 Phase 1(预期 ~30 min)
- 选 B: 重新设计,Phase 0 重做(预期 ~3-4 hour)
- 选 C: 放弃双码本路线(等于判 Phase 2 NO-GO)

---

## 6. 关键决策点回顾

| 决策 | 选择 | 理由 |
|------|------|------|
| α_geo = 0.1 | 试过 | 不够,geo_loss 仍发散 |
| τ = 0.5 软分配温度 | 试过 | softmax(-d/0.5) 仍尖锐,代偿 argmin 不可微不足 |
| z-mean centering for L0 | 试过 | Phase 0 v3 PASS,但 Phase 1 暴露 latent 范数 = 0 的更深层 bug |
| kill Phase 1 (epoch 46) | ✅ | 100% 已知失败,继续只浪费 GPU |

---

## 7. 产物清单

- `products/task200/dual_arm_C_smoke/` — Phase 1 训练 ckpt(已 kill,但 epoch_39/epoch_44 ckpt 落盘)
- `logs/task200/phase1_arm_C.log` — 完整 50 epoch 日志
- `logs/task200/phase0_metrics_center{False,True}.json` — Phase 0 v2/v3 验证结果
- `verdicts/task200_phase0_result.md` — Phase 0 PASS verdict
- `scripts/task200_phase0_dual_codebook_init.py` — Phase 0 验证脚本
- `scripts/task200_phase1_smoke.sh` — Phase 1 launcher

---

## 8. 后续建议

**当前状态**:Phase 0 PASS / Phase 1 FAIL,Phase 2 阻塞等用户决策。

**§16 backlog**:有 #201(Stage 3 逐层 κ,等 #200)阻塞,需先解 Phase 1。

**下一步(R11 自主决策)**:
- 用户已明确说"马上开始不需要我批准" — 但本任务 Phase 1 fail 后,**设计层改动超出 R11.5 自主范围**, 必须 dry-run 报告修复路径 + 等用户拍板
- 主动推进 #201/其他 backlog 也都被 #200 阻塞
- GPU 1 还有 #202 Sinkhorn Stage 3 在跑,3 张卡空闲

**建议**:用户拍板修复方向(A/B/C)后,15-30 min 内可重跑 Phase 1。

result: Phase 1 FAIL — 核心 bug 是 dual_codebook 路径下 init_emb 用 proj_to_ball(expmap0(latent)) 把切空间向量强制推到原点(‖latent‖_E=0),导致码本坍缩(collision 96%)。Phase 0 PASS 是因为 standalone 验证未走 HRQVAE.forward 的 logmap0 转换。等用户拍板修复方向(A:加 expmap0/logmap0 转换、B:重构 forward、C:放弃双码本路线)。