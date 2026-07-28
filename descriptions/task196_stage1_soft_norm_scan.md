# Task #196 — Stage 1: 软约束 (切空间范数钉在 r_target)

> **任务目的**: 在 HVectorQuantization.forward 加 norm_loss 软约束, 验证"切空间范数钉在 r_target/2"能否把码字推到目标 ρ (2.0/2.7/3.4).

> **完成日期**: (待启动, 等 Stage 3 #194 完成释放 4 GPU)
> **状态**: 🟡 待启动 (设计完成, 排队等 GPU)

---

## 1. 背景

承接 Stage 0 (#195): 实测 ρ ≪ 目标 ρ, 软约束是必要的第一步. 最小改动方案.

用户 2026-07-25 设计:
```python
# HVectorQuantization.__init__ 里加
self.r_target = r_target          # 1.0 / 1.35 / 1.70, 由外部按层传入
self.gamma = gamma                # 约束权重

# forward 里 loss 之后加
norm_loss = ((codebook.norm(dim=-1) - self.r_target) ** 2).mean()
loss = commitment_loss + self.beta * codebook_loss + self.gamma * norm_loss
```

`HResidualVectorQuantization.__init__` 里按层分发 r_target, 和 n_e_list 一样 zip 进去.

---

## 2. 实验设计

**变量**: γ ∈ {0, 0.1, 0.5, 2.0} (4 臂)
**保持不变**: Task #194 Stage 1 recipe (K0=64 baseline, num_emb_list=[64,128,256], e_dim=32, β=0.5)
**启动命令**: 4 臂并发, 每臂 1 张 GPU

```bash
CUDA_VISIBLE_DEVICES=$GPU python3 -u $REPO/HG-Rec/src/train_hrqvae.py \
    --dataset Instruments --epochs 500 --num_emb_list [64,128,256] \
    --e_dim 32 --beta 0.5 --loss_type poincare \
    --r_target [1.0,1.35,1.70] --gamma $GAMMA --save_limit 50
```

---

## 3. 决策触发 (按优先级)

| 检查 | 通过标准 |
|------|----------|
| 半径是否被推上去 | rho_p50 三层 ≥ 1.8 / 2.4 / 3.0 |
| **几何是否激活** | **lambda_p50 三层 ≥ 4.0 / 7.0 / 12.0** |
| 码本没崩 | collision ≤ 基线的 1.5 倍 (即 ≤ 13%) |
| 重构没崩 | recon_loss 不发散 |

---

## 4. 预期失败模式

码字被拉到 1.0 (目标 r_target), 但残差还是 0.1 量级 → 码字和残差尺度严重不匹配 → 重构变差、collision 上升.

**如果出现, 不要调参硬扛, 直接进 Stage 2 (#197)** — 那是它的正解.

---

## 5. 预算

| 阶段 | 估算时间 |
|------|---------|
| 4 臂 Stage 1 (500 epoch × 4 GPU 并发) | ~4-6 h |
| 几何诊断 (rho/lambda/方向) | ~10 min |
| verdict aggregate | ~10 min |
| 总计 | ~5 h |

---

## 6. 风险与缓解

**风险 1**: γ 太小 (0.1) 拉不动码字 → 缓解: 直接用 γ=0.5/2.0
**风险 2**: γ 太大 (2.0) 主导 loss → 缓解: 检查 recon_loss 是否发散
**风险 3**: kmeans_init 初始化的码字范数 ≠ r_target → 缓解: 训练前预归一化到 r_target

---

## 7. 完成度跟踪

- [ ] Stage 0 完成后微调 r_target (基于实测 ρ)
- [ ] 4 臂启动 (γ = 0, 0.1, 0.5, 2.0)
- [ ] 每臂跑完 500 epoch + collision + lambda 监控
- [ ] verdict 写盘 (verdicts/task196_stage1_soft_norm_result.md)
- [ ] 决定是否进 Stage 2 (#197)

---

## 8. 与已有任务的关系

| Task | 关系 |
|------|------|
| #194 Stage 1 (500 epoch, 4 K0) | 提供 baseline 几何参照 |
| #195 Stage 0 | 提供实测 ρ vs 目标 ρ 差距 |
| #197 Stage 2 | 失败模式后的正解 |

---

## 9. 后续动作

- **GO (几何激活 + 性能不掉)**: 进 Stage 2 (#197)
- **NO-GO (γ 拉到目标但 collision/recon 崩)**: 进 Stage 2 (#197) — 不要调参硬扛
- **NO-GO (γ 完全拉不动码字)**: 检查 r_target 是否过高, 调小目标 ρ 后再扫