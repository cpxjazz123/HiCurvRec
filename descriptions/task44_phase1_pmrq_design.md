# Task #44 — Phase 1 PM-RQ 完整实验设计 (3 组对照 + 1 消融)

> **任务目的**: 用三层对照 (T5-baseline < MCKG-standard < PM-RQ) + 一个双曲必要性消融,验证可学习三段混合曲率残差量化是否带来标准欧氏 RQ-VAE 无法达到的重建/下游收益,并通过训练轨迹观察双曲段权重是否自然收敛到低值(交叉验证 Task #42)

> **完成日期**: (in progress)
> **状态**: 🟡 Phase 1 设计已就绪,待启动

---

## 1. 背景

承接:
- **Task #36**: fused_64d gap=+0.0307 STRONG (per-item protocol, 99% CI [+0.0264, +0.0353])。MCKG embedding 空间携带 next-item 信号
- **Task #42**: Riemannian 重测发现 subspace_2_hyperbolic 用真正的双曲距离仍 WEAK,**意外发现** sphere 段用 Riemannian 比 Euclidean 强 5.2× → **sphere 段必须用 Riemannian, 不能 fallback Euclidean**
- **Task #43**: norm 长尾诊断 — fused + subspace_1 + subspace_2 必须 clip/normalize,否则训练坍缩到离群 item
- **Task #22 phase0 D0**: MCKG 训练收敛后 κ 实际值 = [+0.8446, -0.1741, -1.0586](注意:这是 D0 诊断值, MCKG entity_embedding.pt 里的 kappas=[+5.05, -0.08, -5.04] 是初始设定,非收敛值)

**用户 2026-07-20 关键修正**:
1. **PM-RQ 输入必须是 3 段原始子空间,不是 fused_64d** (fused_64d 已被 log map+平均 flatten)
2. **融合权重必须中性初始化**(softmax([1,1,1])=[1/3,1/3,1/3]),**不预判** hyperbolic 会被压低 (Task #42 是独立验证,不能影响初始化)
3. **可学习 κ 用 D0 诊断值初始化** = [+0.8446, -0.1741, -1.0586]
4. **独立的双曲必要性消融**(D 组 = 去掉 hyperbolic 的两段 PM-RQ),看 hyperbolic 是否真的可以砍掉

---

## 2. 实验组设计

| 组别 | 名称 | 输入 | 量化方式 | 验证内容 |
|---|---|---|---|---|
| **A** | T5-baseline | T5 768d (sentence-t5-base) | 标准欧氏 RQ-VAE | TIGER 原始 baseline 复现 |
| **B** | MCKG-standard | fused_64d (L2-normalized) | 标准欧氏 RQ-VAE | 隔离"仅换 embedding 来源"的价值 |
| **C** | **PM-RQ (3 段)** | 3 段原始子空间 (各 64d) | PM-RQ (可学习 κ + 可学习权重) | **核心方法** |
| **D** | PM-RQ (2 段消融) | 只 sphere + euclid | PM-RQ 去 hyperbolic 分支 | **验证双曲段是否可砍掉** |

**Recall 链条**: A < B < C, **且 D ≈ C**(双曲确实贡献小) → 最干净的结果结构

---

## 3. 实验 0 — 前置 bug 修复与 norm 诊断 (必须先做)

### 0.1 修复 dist_kappa 在 κ=0 的退化 bug

```python
def test_dist_kappa_at_zero():
    """κ → 0 时 dist_kappa 必须收敛到标准欧氏距离."""
    x = torch.randn(10, 64)
    y = torch.randn(10, 64)
    kappa = torch.tensor(1e-6)
    d_kappa = dist_kappa(x, y, kappa)
    d_euclid = torch.norm(x - y, dim=-1)
    assert torch.allclose(d_kappa, d_euclid, atol=1e-3), "κ→0 退化检验失败"
```

不修这个,**训练中 κ 一旦被梯度推向 0 附近,loss 直接产生垃圾数值**。Task #42 sanity 已暴露此问题(fused_64d κ=0 算出 gap=-0.28,完全不对)。

### 0.2 norm 诊断 + clip (per-subspace)

```python
# 对每个子空间(不是 fused 后的)分别看 norm 分布
for name, subspace in [('sphere', s), ('euclid', e), ('hyperbolic', h)]:
    norms = subspace.norm(dim=-1)
    print(f"{name}: mean={norms.mean():.3f}, std={norms.std():.3f}, "
          f"p99={norms.quantile(0.99):.3f}, max={norms.max():.3f}")

# item-wise clip (不标准化到单位球——保留"热度梯度"这个之前诊断出的有价值信号)
norm_cap = 5.0  # 由诊断决定 (Task #43 给出: sphere p99=0.57, euclid p99=4.79, hyperbolic p99=10.20)
clipped = subspace * torch.clamp(norm_cap / subspace.norm(dim=-1, keepdim=True), max=1.0)
```

**与 Task #43 差异**: Task #43 用 p99 作为 clip 阈值,本设计用固定 norm_cap=5.0 作为统一上限(因为不同 subspace p99 差异大,统一 clip 更公平)。

---

## 4. 实验 1 — 输入准备 (3 段原始子空间)

```python
# 正确输入: 三个各自 64 维、各自留在原生流形上的子空间
subspace_sphere      # (11924, 64), 原生球面表示,配 κ_sphere
subspace_euclid      # (11924, 64), 原生欧氏表示,κ 固定 = 0
subspace_hyperbolic  # (11924, 64), 原生双曲表示,配 κ_hyperbolic

# 不使用 fused_64d — 它在 log map + 平均这一步丢失了可分离的曲率结构
```

**norm clip** (per-subspace, norm_cap=5.0):
- sphere: clip 后保证不破坏"角度"信号
- euclid: 直接 clip
- hyperbolic: clip 后保留 Lorentz norm 比例

---

## 5. 实验 2 — 核心模型: PM-RQ 本体

```python
class PMRQ(nn.Module):
    def __init__(self, K=256, dim=64, n_layers=1):
        super().__init__()
        # κ: D0 诊断值初始化 (唯一允许带先验的地方)
        self.kappa_sphere = nn.Parameter(torch.tensor(0.8446))
        self.kappa_hyperbolic = nn.Parameter(torch.tensor(-1.0586))
        # 欧氏 κ 固定为 0, 不学

        # 融合权重: 中性初始化, 不带 Task #42 的偏见
        self.fusion_logits = nn.Parameter(torch.ones(3))  # softmax([1,1,1]) = [1/3,1/3,1/3]

        # 三个码本, 容量一致, 不预先厚此薄彼
        self.codebook_sphere = nn.Parameter(torch.randn(K, dim) * 0.01)
        self.codebook_euclid = nn.Parameter(torch.randn(K, dim) * 0.01)
        self.codebook_hyperbolic = nn.Parameter(torch.randn(K, dim) * 0.01)

    def forward(self, r_s, r_e, r_h):
        d_s = geodesic_distance_sphere(r_s, self.codebook_sphere, self.kappa_sphere)
        d_e = euclidean_distance(r_e, self.codebook_euclid)
        d_h = lorentz_distance(r_h, self.codebook_hyperbolic, self.kappa_hyperbolic)

        w = F.softmax(self.fusion_logits, dim=0)
        d_total = w[0]*d_s + w[1]*d_e + w[2]*d_h

        idx = d_total.argmin(dim=-1)  # 单一联合索引 i*, 见此前 settled 设计
        c_s, c_e, c_h = (self.codebook_sphere[idx],
                         self.codebook_euclid[idx],
                         self.codebook_hyperbolic[idx])

        # 残差更新: Scheme B, component-wise log/exp
        r_s_next = exp_map(c_s, -log_map(c_s, r_s, self.kappa_sphere), self.kappa_sphere)
        r_e_next = r_e - c_e
        r_h_next = exp_map(c_h, -log_map(c_h, r_h, self.kappa_hyperbolic), self.kappa_hyperbolic)

        return idx, (c_s, c_e, c_h), (r_s_next, r_e_next, r_h_next)
```

**Loss** (per settled design, reconstruction only):
```python
loss = mse(r_s, c_s) + mse(r_e, c_e) + mse(r_h, c_h)
```

---

## 6. 监控指标 (贯穿训练, 不只看终值)

```python
trajectory = {
    'step': [],
    'kappa_sphere': [], 'kappa_hyperbolic': [],
    'w_sphere': [], 'w_euclid': [], 'w_hyperbolic': [],  # 核心:看 w_hyperbolic 是否自然下降
    'codebook_utilization_sphere': [],   # 每段各自的 collapse rate
    'codebook_utilization_euclid': [],
    'codebook_utilization_hyperbolic': [],
    'recon_mse': [],
}
```

**每 500 step 记录一次**,训练结束后画出 `w_hyperbolic` 随时间的曲线 — 这条曲线本身就是"训练自然发现了什么"的直接证据。

---

## 7. Go/No-Go 判据

### Phase 1 (Toy, 10K items 子集, K=64) 成功标准

1. 三个 κ 训练中不发散、不产生 NaN(验证 0.1 的 bug 修复生效)
2. 三段 codebook utilization 都不为 0(没有整段坍缩)
3. **Recon MSE: C 组 ≤ B 组**(混合曲率至少不比标准量化差)

### Phase 2 (全量 Toys, 接 TIGER 下游) 判据

- **核心假设成立**: C 组 R@5 > B 组 R@5 > A 组 R@5
- **双曲必要性判据**: C 组 vs D 组,若 D 组 R@5 与 C 组统计不可区分(差距<0.5pp),且训练中 `w_hyperbolic` 收敛到 <0.15 → 说明可以用更简单的两段设计替代
- **交叉验证判据**: `w_hyperbolic` 收敛值,与 Task #42 的 gap 排序 (sphere > euclid >> hyperbolic) 方向一致 → 两个独立信号源的交叉确认

---

## 8. 时间预算

| 阶段 | 估算 |
|---|---|
| 实验 0 (bug 修复 + norm 诊断) | 1-2 天 |
| 实验 1-2 (PM-RQ 实现 + 单元测试) | 3-4 天 |
| Phase 1 toy (4 组 × 10K items) | 3-5 天 (可并行) |
| Phase 2 (全量 + TIGER 下游) | 1-2 周 |

---

## 9. 与之前 Task 的差异

| 之前 | 现在 (Task #44) | 原因 |
|---|---|---|
| Task #23/#24 PM-RQ × TIGER (fused_64d 输入) | **3 段原始输入** | fused 已 flatten, 几何丢失 |
| Task #38 prototype (fused → 标准 RQ-VAE) | **B 组保留,作为对照** | 隔离"仅换 embedding"贡献 |
| κ 随机初始化 | **D0 诊断值初始化** | 利用已知的几何先验 |
| 权重手设 (Task #41 提议 sphere>euclid>>hyperbolic) | **中性初始化 [1/3,1/3,1/3]** | 不预判, 让训练说话 |
| 没有 hyperbolic 消融 | **D 组 (2 段消融)** | 独立验证双曲是否可砍掉 |
| 没有训练轨迹监控 | **每 500 step 记录** | 让 "w_hyperbolic 自然下降" 可见可验证 |

---

## 10. 完成度跟踪

- [ ] 实验 0.1 dist_kappa κ=0 退化 bug 修复
- [ ] 实验 0.2 norm clip (per-subspace, norm_cap=5.0)
- [ ] 实验 1 输入准备 (3 段原始 + clip + 验证 shape)
- [ ] 实验 2 PMRQ 类实现 + 单元测试 (κ 不发散 / utilization 不为 0 / recon_mse 下降)
- [ ] Phase 1 toy: A 组 T5-baseline (10K items, K=64)
- [ ] Phase 1 toy: B 组 MCKG-standard (fused_64d, K=64)
- [ ] Phase 1 toy: C 组 PM-RQ 3 段 (K=64)
- [ ] Phase 1 toy: D 组 PM-RQ 2 段 (sphere+euclid, K=64)
- [ ] Phase 1 决策: 是否进入 Phase 2
- [ ] (Phase 2) 全量 Toys + TIGER 下游 4 组

---

**核心设计特点**: 所有之前反复讨论确立的原则都被具体化成了代码里的一行——中性初始化(不是 Task #42 的偏见)、D0 只做初始化(不做锁死)、独立的双曲必要性消融(不预判)、全程轨迹监控(让"自然学到"这件事可见可验证)。
