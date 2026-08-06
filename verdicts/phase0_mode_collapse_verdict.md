---
type: verdict
created: 2026-08-02
tags:
  - phase0
up: "[[index]]"
---
# Phase 0 fix 隐藏的 85% RQ-VAE mode collapse (用户 2026-07-25 第二轮诊断)

> **核心 falsification**: 用户指认的 "λ=2万" 来自 **task180_graph_aware (200 epoch, Phase 0 fix)**, 不是 task181 (50 epoch). **task181 是 50 epoch 短训** — 范数还没涨到边界 (mean ‖x‖_E=0.26, q95=0.31). **task180/task178 是 200 epoch 长训** — 范数已被推到 boundary (mean 0.85-0.90, q95=1.0000). 我之前用 task181 的数字证明"饱和假设 falsified" 是**诊断对象错**.
>
> 更严重: task178/task180 的 200 epoch RQ-VAE 训练, **collision_rate 飙到 85.24% / 81.70%**, **reconstruction_loss 卡死在 1489.5494** (9 个 epoch 一字不变). 这是 **Phase 0 fix 自己引入的 mode collapse** — Poincaré distance + β=0.5 + paper convention + 200 epoch 长训 = 码字全推 boundary + encoder 卡死. **Phase 0 fix 不只没修好, 还退化了**.

---

## 1. 用户的两条 falsification

### Falsification 1: λ=2万 是 task180, 不是 task181

**确认**: 我上一轮诊断脚本 `scripts/diagnose_cnorm_distribution.py` 加载的是 task181 (50 epoch Phase 0 fix). task181 当时 mean ‖x‖_E ≈ 0.26 — 离 boundary 还远. 所以我才敢说 "饱和假设 falsified".

**真相**: λ=2万 来自 **task180_graph_aware** (200 epoch, Phase 0 fix + LightGCN). task180 是 graph-aware RQ-VAE 的 SID 来源, 不是 baseline. task180 是 "#180 的 Phase 3 Stage 1 RQ-VAE", 我以为是 "对比 baseline" — **诊断对象搞错了**.

### Falsification 2: task180 数字自相矛盾

**确认**: task180 L0:
- eucl_norm_mean = 0.8991 (用户记得的 "0.85" 数字)
- eucl_norm_q95 = 1.0000 (95% 码字在 boundary)
- eucl_norm_max = 0.9999989867 (= 1 - 1e-6, 这是 `proj_to_ball` 的 maxnorm ceiling)
- conformal_mean = 18822.56 (不是 2万, 是 1.88万, 跟用户记的 2 万一致)
- conformal_max = 200000.0 (clamp ceiling)

**计算验证**: ‖x‖_E mean=0.899 → κ‖x‖²=0.808 → λ=2/(1-0.808)=**10.4** (不是 18822). 矛盾.
**正解**: λ=18822 来自 q95 那些 ‖x‖=1.0 的码字 (κ‖x‖²=1.0 → 1-1.0=0 → 除零 → clamp 到 200000). 所以 mean λ 18822 是被 clamp ceiling 拉高的, 不是 ‖x‖=0.899 算出来的.

**我之前漏掉的**: 只看了 mean, 没看 q95. q95=1.0000 就是边界饱和的明确证据.

---

## 2. 新发现: Phase 0 fix 自己引入 mode collapse

对比 task178 (Poincaré, Phase 0 fix, 200 epoch) 和 task179 (Euclidean, β=0.25, 200 epoch) 的 RQ-VAE 训练日志:

| 任务 | Recipe | Epoch | collision_rate (epoch 199) | recon_loss (stuck since) |
|------|--------|-------|---------------------------|--------------------------|
| **task178** | Phase 0 fix + β=0.5 + **Poincaré** + Sinkhorn | 200 | **85.24%** | **1489.5494 (9 epoch 卡死)** |
| **task180** | Phase 0 fix + graph + β=0.5 + **Poincaré** + Sinkhorn | 200 | **81.70%** | 1489.5494 (9 epoch 卡死) |
| **task179** | β=0.25 + **Euclidean** + Sinkhorn | 200 | **20.79%** | **0.0021 (干净)** |
| **task181** | Phase 0 fix + β=0.5 + **Poincaré** + Sinkhorn | **50** (短训) | **5.39%** | 8.9164 (仍下降) |

**Smoking gun**:
- task178 和 task180 的 recon_loss 卡死在 **1489.5494** (epoch 190 → 199 全部相同, 字面相等) — **encoder 输出的 MSE 没变** — **encoder 已经不学了**, 整个数据集被映射到同一个点.
- task179 (Euclidean) recon_loss 0.0021, **小 5 个数量级** — encoder 正常工作.
- task181 50 epoch 时 recon_loss 8.92 (仍下降), collision 5.39% — 还没崩, 但训到 200 epoch 会跟 task178/task180 一样.

**根因分析** (从代码逻辑推):

```python
# task178 RQ-VAE forward (Phase 0 修过的):
latent_h = proj_to_ball(expmap0(latent, c))   # 把欧式 latent 映到双曲
codebook_h = proj_to_ball(expmap0(codebook, c))
commitment_loss = poincare_distance(x_q_h.detach(), latent_h, c) ** 2  # 编码器梯度
codebook_loss = poincare_distance(x_q_h, latent_h.detach(), c) ** 2     # codebook 梯度
loss = codebook_loss + beta * commitment_loss  # beta=0.5 (paper convention)
```

Poincaré distance 在 boundary 附近**趋近 ∞**, 梯度∝1/(1-c‖x‖²)² — boundary 附近梯度饱和. 这意味着:
1. **codebook 被推到 boundary**: 离输入 latent 最近的 codebook 候选要尽量覆盖 latent → 把 codebook 推到 boundary 外附近, 因为 boundary 那边 Poincaré 距离最大.
2. **encoder 卡死**: ‖x‖=1.0 时 `1 - c‖x‖² = 0`, 反向传播 `∂L/∂latent` 数值不稳 → encoder 学不动 → recon_loss 卡死.
3. **Sinkhorn at Stage 2 制造 SID 唯一性假象**: Sinkhorn 强制 balanced assignment, 即使 underlying codebook 已坍缩, 也输出 9922/9922 unique SIDs. **Sinkhorn 把 mode collapse 包装成"成功"**.

所以 task181 (50 epoch) 看起来 "util=98.4%, collision=5.4%" 是**虚假健康** — 50 epoch 不够长, 还没到崩塌点. task178 (200 epoch) 是真崩塌后的样子.

---

## 3. Stage 3 R@10 数字的解读 (重要)

任务仍在跑的 #178/#179/#180 Stage 3 各自的 R@10:

| Task | Recipe (Stage 1 RQ-VAE) | Stage 1 collision | SID tensor | Stage 3 R@10 |
|------|--------------------------|--------------------|------------|--------------|
| #178 | Phase 0 fix + Poincaré (崩) | **85.24%** | Sinkhorn 强制 unique | **0.1135** (best!) |
| #179 | Euclidean + Sinkhorn | 20.79% | Sinkhorn 强制 unique | 0.0893 |
| #180 | Phase 0 fix + graph + Poincaré (崩) | 81.70% | Sinkhorn 强制 unique | 0.1049 |

**关键悖论**: task178 (Stage 1 collision 85%!) Stage 3 R@10 = 0.1135, **最高**. 旧 baseline task84 R@10 = 0.1020, 反而更低.

**两种解读**:
- **(a) Sinkhorn balanced assignment 给了 T5 比"自然 VQ 唯一化"更好的离散 token 序列** — 哪怕 underlying codebook 坍缩, Sinkhorn 旋转后产生的 SID 序列恰好适合 T5 自回归预测.
- **(b) T5 5.5M 直接 memorize 了 item→SID 的 (Sinkhorn-induced) 任意映射** — 跟 codebook 几何质量无关, 只跟"输入离散 token, 输出离散 token"的格式有关.

**两种解读都意味着**: 当前 #178 的 R@10=0.1135 **不能用来支持 "Phase 0 fix 修复了 HG-Rec" 的结论**. 它只是 "T5 在 9922 项 Sinkhorn-balanced SID 上训练到 200 epoch 能达到 0.1135" 的纯 T5 学习能力 demo.

**但**: 用户指令"先别叠"的意思是 — **不要把这个 0.1135 当成 baseline 之上去搭 #179 双分支 / #180 graph-aware**. 因为:
1. 双分支叠加在 degenerate SID 上 → 实验无法区分 "双分支有用" 还是 "Sinkhorn-balanced SID + T5 有用"
2. graph-aware Stage 1 同样崩到 81.7% collision → 单变量对照失效 (两个变量都坏了, 没法说谁的影响)

---

## 4. 决策选项 (R11.4 不可逆 — 跟用户确认后再执行)

### Option A: Kill #178/#179/#180 Stage 3
- **理由**: SID foundation 已崩. Stage 3 R@10 不可解读, 不值得继续烧 GPU.
- **代价**: 浪费了 ~50 min Stage 3 训练 (每个 ~0.5-1h)
- **收益**: 释放 GPU 0/1/2, 给 Phase 0 重设计留出空间.
- **R11.3 推荐**: ✅ 推荐. 用户明确说"先别叠".

### Option B: 让 #178/#179/#180 Stage 3 跑完
- **理由**: R@10=0.1135/0.0893/0.1049 仍然有意义 (作为 "T5 + degenerate SID" 的 baseline)
- **代价**: ~2-4 小时 GPU 时间
- **收益**: 拿到 Stage 4 R@10 数字, 即使几何解读错误, T5 学习能力的对比仍可作 "T5 in different Sinkhorn-balanced SID regimes" 的 appendix
- **R11.3 不推荐**: 用户明确说"先别叠", 继续等于在崩的基础上继续堆.

### Option C: Phase 0 重设计 (技术修复)
不杀进程, 同步启动 Phase 0 re-design:
1. 把 commitment loss 从 `poincare_distance` 换成 `MSE` (Euclidean)
2. codebook 单独用 `poincare_distance` 训练
3. 加 `||x||² ≤ α` 正则化 (α≈0.5) 防止 boundary 饱和
4. 重新跑 50/100/200 epoch 看 collision_rate 演化

**R11.3 推荐**: C 在用户确认后再启动, 因为修改 HG-Rec 上游代码需要 EnterPlanMode + 用户批准.

---

## 5. 即将采取的 R11.3 自主决策 (这次)

1. ✅ 立即更新 #178/#179/#180 task 描述 (Stage 3 在 degenerate SID 上跑, 数字不可解读为几何优势)
2. ✅ 写本 verdict (完成)
3. ✅ 更新 memory: 新增 "phase0-mode-collapse" pointer
4. ⏸️ **不动 GPU 0/1/2 Stage 3 进程** — 等用户确认
5. ⏸️ **不启动 #204 embedding 归一化 phase** — 跟地基修复一起做 (Option C)
6. ⏸️ **更新 CLAUDE.md baseline 数字 (从 0.1020 → 0.1135) 推迟** — 因为 0.1135 不是干净的 baseline, 是 Sinkhorn-degenerate-SID + T5 学习能力

---

## 6. 产物路径

- 数据: 无新产物 (诊断重用 `products/codebook_hypnorm_diagnostic.json`)
- 脚本: `scripts/diagnose_cnorm_distribution.py` (加载的 ckpt 范围是对的, 但解释需要重写)
- Verdict: `verdicts/phase0_mode_collapse_verdict.md` (本文件)
- Memory pointer: 新增 `memory/phase0-mode-collapse.md`

---

## 7. 用户待决策

| 选项 | 内容 | 推荐 |
|------|------|------|
| A | Kill #178/#179/#180 Stage 3, 释放 GPU 给 Phase 0 重设计 | ✅ |
| B | 让 Stage 3 跑完, 收 R@10 作为 "T5 + Sinkhorn-SID" 数据点 | ❌ |
| C | 同 A, 同步启动 Phase 0 重设计 (commitment MSE + norm 正则化) | ✅ + 修复 |

**建议回复**: "A" 或 "C" 任选. C 更彻底, A 更保守 (只杀进程, 不动代码).