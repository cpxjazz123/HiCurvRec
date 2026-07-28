# Task #197 — Stage 2: 双码本解耦 (分配路径 + 重构路径)

> **任务目的**: 解决 Stage 1 软约束的失败模式 — 同一组码字既要负责"选哪个"又要负责"减多少", 尺度相反. 解法: 每层两个码本, 共享 index.

> **完成日期**: (待 Stage 1 #196 通过后启动)
> **状态**: 🟡 待启动 (设计完成, 排队等 Stage 1 verdict)

---

## 1. 背景

承接 Stage 1 (#196) 预期失败模式: 码字被拉到目标 ρ 但残差量级不匹配.

**根本问题**: 同一组码字既要负责"选哪个"又要负责"减多少", 而这两件事要求的尺度相反.
- 分配 (决定 SID) → 双曲几何需要码字在大半径
- 重构 (决定残差级联) → 残差量级天然小, 码字不该大

**解法**: 每层两个码本, 共享同一个 index.
- `emb_geo`: 钉在 r_target, 只管分配
- `emb_rec`: 无约束, 只管残差 (欧式 MSE)

---

## 2. 实验设计

**核心代码**:
```python
class HVectorQuantization(nn.Module):
    def __init__(self, n_e, e_dim, r_target, ...):
        self.emb_geo = nn.Embedding(n_e, e_dim)   # 几何码本
        self.emb_rec = nn.Embedding(n_e, e_dim)   # 重构码本
        self.r_target = r_target

    def forward(self, x, use_sk=True):
        latent = x.view(-1, self.e_dim)
        c = self.c

        # 分配路径: 纯方向比较, 在目标半径上
        z_dir  = F.normalize(latent, dim=-1)
        z_geo  = self.r_target * z_dir
        cb_geo = self.r_target * F.normalize(self.emb_geo.weight, dim=-1)

        z_h  = expmap0(z_geo,  c)
        cb_h = expmap0(cb_geo, c)
        d = poincare_distance(z_h.unsqueeze(1), cb_h.unsqueeze(0), c).squeeze(-1)
        indices = torch.argmin(d, dim=-1)

        # 几何 loss (在钉住半径的表示上算)
        e_geo = cb_geo.index_select(0, indices)
        e_h   = expmap0(e_geo, c)
        commitment = torch.mean(poincare_distance(e_h.detach(), z_h, c) ** 2)
        codebook   = torch.mean(poincare_distance(e_h, z_h.detach(), c) ** 2)

        # 重构路径: 独立码本, 欧式残差
        x_q = self.emb_rec(indices)
        recon_align = F.mse_loss(x_q, latent.detach())

        loss = commitment + self.beta * codebook + self.alpha * recon_align
        x_q = x + (x_q - x).detach()
        return x_q, loss, indices
```

**关键风险**: kmeans_init 必须改成"方向 kmeans" (latent 先 normalize 再聚类). 直接欧式 kmeans 会因半径不同把码字都拉到 r 大的方向.

---

## 3. 决策触发 (按优先级)

| 检查 | 通过标准 |
|------|----------|
| λ 进窗口 | lambda_p50 ≥ 4.8 / 8.0 / 15.0 |
| collision | ≤ 12% (基线 8.6–12.4%, 持平即算成功) |
| test R@10 | ≥ 0.100 (基线 0.1033, 不掉即算成功) |
| 层内方向分散度 | 每层码字方向的平均夹角 — 太集中说明方向坍缩 |

---

## 4. 重要: 成功标准不是刷分

> Stage 2 的成功标准是"几何激活 + 性能不掉", **不是刷分**.

因为已经用 Task #188 证明了下游性能对码本质量不敏感 — **大幅提升本来就不该被期待**. 
论文的贡献是"让一个从未生效的机制第一次生效, 并首次能够测量它", 而不是 SOTA.

把这一点提前想清楚, 避免 Stage 2 出来性能持平时误判为失败.

---

## 5. 预算

| 阶段 | 估算时间 |
|------|---------|
| 4 臂 Stage 2 (500 epoch × 4 GPU) | ~5 h |
| Stage 3 SID 推断 (Sinkhorn-on) | ~10 min |
| Stage 3 T5-mini (200 epoch × 4 GPU) | ~80 min |
| Stage 4 test R@10 | ~5 min |
| verdict | ~10 min |
| 总计 | ~6 h + ~80 min |

---

## 6. 风险与缓解

**风险 1**: 方向 kmeans 实现错误, 码字全坍缩 → 缓解: 预归一化 + 检查 λ
**风险 2**: 双码本参数翻倍, 过拟合 → 缓解: 码本小 (K=32-256), 实际参数增量可忽略
**风险 3**: emb_geo 和 emb_rec 解耦后训练不稳定 → 缓解: emb_geo 钉死 r_target, 不让梯度流向其 norm

---

## 7. 完成度跟踪

- [ ] HVectorQuantization 双码本化 (rewrite utils.py:179+)
- [ ] 方向 kmeans_init (rewrite utils.py:131+)
- [ ] HResidualVectorQuantization 分发 r_target
- [ ] 4 臂 Stage 1 (500 epoch × 4 GPU)
- [ ] Sinkhorn-on SID 推断 (用 task195 修复过的脚本)
- [ ] Stage 3 T5-mini (1 seed/臂)
- [ ] Stage 4 test R@10
- [ ] verdict 写盘 (verdicts/task197_stage2_dual_codebook_result.md)
- [ ] 决定是否进 Stage 3 (#198)

---

## 8. 与已有任务的关系

| Task | 关系 |
|------|------|
| #195 Stage 0 | 提供 ρ_target 实测基线 |
| #196 Stage 1 | 失败模式的来源, Stage 2 是正解 |
| #198 Stage 3 | Stage 2 通过后才启动 (逐层可学习 κ) |
| #188 paper Table 7 | 证明下游性能不敏感, Stage 2 评估标准参考 |

---

## 9. 后续动作

- **GO (λ 进窗口 + collision ≤ 12% + test R@10 ≥ 0.100)**: 进 Stage 3 (#198)
- **PARTIAL (λ 进窗口但 collision 上升)**: 调 α (recon_align 权重), 或调小 emb_rec 容量
- **NO-GO (码字方向坍缩)**: 检查方向 kmeans 实现, 修正后重跑