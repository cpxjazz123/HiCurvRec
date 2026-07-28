# Task #217 — 方向多样性正则 (第 8 方向, 针对 C1 利用率)

**日期**: 2026-07-26
**父级**: 用户 2026-07-26 反馈 — 共同根因是 √c·ρ 张力 (不是坍缩), A3 (距离饱和) vs C1 (死码本) 是两个独立机制.
**优先级**: 🟠 紧急 (用户明确优先级 3, 但需硬止损)

---

## 1. 背景与动机 (用户 2026-07-26 反馈)

### 1.1 错误叙事纠正

之前的"共同根因" 描述 — "HG-Rec baseline 码字 norm ≈ 1.0, λ_κ ≈ 100, 几何坍缩" — 是**错的**.
真实数据:
- 官方 baseline (Task #84 / #189 / #191): ‖p‖_E = **0.262 / 0.100 / 0.072**, λ_κ = **2.15 / 2.02 / 2.01**
- 边界饱和 (‖p‖_E ≈ 1.0, λ_κ ≈ 2万) 是 **task178 (Phase 0.5 修复)** 的 bug, 不是 baseline 性质

### 1.2 正确的共同根因 — √c·ρ 张力 (定理式)

**无量纲激活参数**: α := √c · ρ
- 官方 baseline: α ≈ 0.54 ≪ 2 (激活阈值), 几何不参与分配 (一致率 99.91%)
- 推入激活区 (α ≳ 2): 几何激活**与量化可分性直接冲突**, 触发两个独立机制:
  - **(A) 距离饱和** (Task #209 A3, 高维钉半径): dyn 2.14 → 1.27, 球面单层薄壳, collision 99.97%
  - **(B) 死码本螺旋** (Task #211 C1, 低维钉半径): L0 利用率 23.4% (15/64)

### 1.3 C1 死因 (本任务针对)

**机制**:
1. 钉半径后, 分配退化为纯余弦 (‖z - e‖² = ‖z‖² + ‖e‖² - 2 z·e, ‖e‖ 钉住, 所以 = const - 2 z·e)
2. 4 维双曲方向分布集中 (Phase 0 已知 dyn 6.5/8.8/12.7 全过 — 方向空间本身不饱和)
3. **重构只用欧式那 32 维** (encoder 输出后 split, 双曲部分不进入 MSE loss)
4. 唯一梯度来自分配损失, 只喂给被选中的 15 个码字 → 49 个码字永远没梯度 → **死码本螺旋**

**Phase 0 已知**: hyp_dim=4 + ρ=[2.0,2.7,3.4] 方向空间 dyn 都过关 (6.5/8.8/12.7), 不是几何问题, 是**梯度流问题**.

---

## 2. 方向多样性正则 (用户提议)

### 2.1 两个互补 formulation

```python
# (a) Gram 矩阵去相关 (强制码字方向散开)
dirs = F.normalize(latent_hyp, dim=-1)     # (B, hyp_dim)
gram = dirs @ dirs.t()                       # (B, B)
L_div = (gram - torch.eye(B)).pow(2).mean()  # 偏离单位矩阵的均方
loss = loss + w_div * L_div

# (b) 分配熵最大化 (强制分配均匀)
p = torch.bincount(idx, minlength=K).float() / len(idx)
L_ent = -(p * (p + 1e-9).log()).sum()
loss = loss - w_ent * L_ent
```

### 2.2 3 臂设计

| 臂 | 架构 | 加的正则 | 目的 |
|---|---|---|---|
| D1 | low-dim product_manifold + 钉半径 (跟 C1 一样) | L_div (w=1.0) | 强制双曲 4 维方向散开 |
| D2 | 同上 | L_ent (w=1.0) | 强制分配均匀 (针对 15/64 利用率) |
| D3 | 同上 | L_div + L_ent (各 w=0.5) | 两个互补 formulation 一起 |

**对照**: Task #211 C1 (low-dim product_manifold + 钉半径, 无方向多样性正则) — 已知 L0 利用率 23.4%, R@10=0.0816

---

## 3. 4 阶段闭环 (3 臂 × 4 阶段 = 12 GPU runs)

| 阶段 | 输出 | 时间 (L40S) |
|---|---|---|
| Stage 1 train_hrqvae.py | HRQ-VAE ckpt + collision | 1-2 小时/臂 |
| Stage 2 SID 推断 (Sinkhorn 30 轮 + dedup) | SID .npy (9922, 4) | 10 min/臂 |
| Stage 3 T5-mini 训练 (R12 强制 ckpt) | T5-mini ckpt | 1.5-2 小时/臂 |
| Stage 4 test eval | R@10 / NDCG@10 | 10 min/臂 |

**总预期**: 3-6 小时 GPU (3 臂 × 4 阶段). 串行 (避免 GPU 抢占).

---

## 4. 判据 (gate)

| 指标 | 通过 | 失败 |
|---|---|---|
| L0 utilization | ≥ 90% | < 90% |
| L1 utilization | ≥ 90% | < 90% |
| Stage 4 test R@10 | ≥ 0.1020 (baseline) | < 0.1020 |
| Stage 4 test NDCG@10 | ≥ 0.0755 (baseline) | < 0.0755 |

---

## 5. **硬止损 (用户 2026-07-26 明示)**

> "如果加上方向多样性正则后, L0 利用率仍然 < 90%, 或利用率上去了但 R@10 仍 < 0.1020 — **收线, 不再试第 9 个**"
> "理由: 利用率是这次唯一未被针对性处理的失效指标. 它一旦被处理还是不行, 就说明问题不在任何单一环节, 而在那个 √c·ρ 张力本身 — 而那个已经被证明是结构性的"

**触发任一即收线**:
- (i) L0 utilization 仍 < 90%
- (ii) utilization ≥ 90% 但 R@10 < 0.1020

**收线后**: 写 verdicts/task217_*.md "第 8 方向 NO-GO, 几何路线 8 方向全部收线, √c·ρ 张力是结构性冲突, 不可解".

---

## 6. 实施步骤

1. 0.5h: 改 train_hrqvae.py (加 --w_div, --w_ent CLI flag, L_div + L_ent loss 计算)
2. 0.5h: 写 task217_D1/D2/D3_stage1_train.sh launcher (跟 task211 C1 类似 + 新 flag)
3. 1-2h/臂 × 3 臂 = 3-6h: Stage 1 训练 (后台)
4. 0.5h: Stage 2 SID 推断 (Sinkhorn 30 轮 + 4th-digit dedup)
5. 1-2h/臂 × 3 臂: Stage 3 T5-mini 训练
6. 0.5h: Stage 4 eval
7. 0.5h: 写 verdict (跟 gate 比, 触发硬止损 → 收线 或 ✅ PASS)

---

## 7. 链接

- 用户反馈原文 (2026-07-26): paper.md §6.7
- 7 方向共同根因: paper.md §6.7.2 (√c·ρ 张力定理式)
- A3 距离饱和: verdicts/task209_path_reg_result.md
- C1 死码本: verdicts/task211_low_dim_pinned_radius_arch_infeasible.md
- baseline ckpt: products/task84/ckpt/Instruments/.../best_loss_model.pth
- 数据集: HG-Rec/dataset/Instruments/item_emb.parquet (9922 × 768)

---

(本文档为 Task #217 完整设计, 不锁定权重 — 实际 w_div / w_ent 在脚本里 sweep.)
