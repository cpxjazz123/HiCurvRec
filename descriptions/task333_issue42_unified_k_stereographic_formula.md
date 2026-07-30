# Task #333 — Issue #42 κ-Stereographic 统一公式实现 (0 GPU, 数学正确性验证)

**日期**: 2026-07-30 14:35
**触发**: Issue #42 owner 2026-07-30 11:55 创建, 用户 2026-07-30 14:34 决策 "先做#42的这个issue"
**状态**: 🟡 **数学实现 + 数值验证, 0 GPU, no training**
**优先级**: R10 主动推进 (曲率相关任务, 跟用户 2026-07-30 14:25 决策一致)
**类型**: 数学层修复 (post-mortem 落地)

---

## 1. Issue #42 核心主张 (来自 owner)

| 项 | 内容 |
|----|------|
| **现状** | R137 用 `torch.where` 3-branch 选择器 (eucl_sq / sph_sq / hyp_sq), **不是** MCKG Table 1 风格的真统一公式 |
| **θ=0 死点根因** | `eucl_sq` 分支数学上不依赖 κ, 即使 R137 修复完全生效, θ=0 init 仍是数学 fixed point (Task #137 verdict 自承) |
| **MCKG Table 1 建议** | 用 `tan_κ / tan_κ⁻¹` 类连续函数, 把球面/双曲/欧氏用同一条数学公式表达, κ 在整个定义域 (含 κ=0) 连续可导 |
| **对称初始化建议** | 替代当前固定 θ_init=+0.01, 用 θ_init=±0.01 / N(0,σ²) |

---

## 2. 当前 R137 实现的实际状况 (新发现)

读完 `HG-Rec/model/hrqvae_free_curv.py:45-76` 后, 实际情况比 Issue #42 描述的更精细:

```python
# R137 实际: 单条 closed-form 公式 (Berman-Metzler 2020 / Chlenski 2020)
def geodesic_distance_sq(x, y, kappa):
    kappa_abs = kappa.abs().clamp(min=1e-8)
    sqrt_kappa = torch.sqrt(kappa_abs)
    diff = x - y
    diff_norm = diff.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    diff_norm_sq = diff_norm ** 2
    denom = 2.0 * (1.0 - kappa * diff_norm_sq / 4.0).abs().clamp_min(1e-6)
    arg = sqrt_kappa * diff_norm / denom
    d = (2.0 / sqrt_kappa) * torch.arctan(arg)
    return d ** 2
```

**这是一个单条 closed-form 公式, 没用 `torch.where`**, 但有 3 处梯度不连续:
1. **`kappa.abs()`** — 在 κ=0 不可导
2. **`(1.0 - kappa * diff_norm_sq / 4.0).abs()`** — 在 κ·r²=4 (antipodal cap) 不可导
3. **`clamp(min=1e-8)`** — 在 clamp 边界不可导

所以 Issue #42 的本质问题 (θ=0 死点) **仍然存在**: R137 把 `eucl_sq/sph_sq/hyp_sq` 三选一换成单公式, 但 `.abs()` 操作本质上在 κ=0 处强制选了"哪个 branch" (因为正负号翻转), 跟 `torch.where` 等价 (只是把硬 if 换成 abs 但仍有分支感).

---

## 3. MCKG Table 1 真统一公式设计

### 3.1 Table 1 公式 (来自 papers/MCKG.md: Table 1)

| 操作 | 公式 |
|------|------|
| **Addition (Möbius)** | $x \oplus_\kappa y = \frac{(1 - 2\kappa\langle x,y\rangle - \kappa\|y\|_2^2)x + (1 + \kappa\|y\|_2^2)y}{1 - 2\kappa\langle x,y\rangle + \kappa^2\|x\|_2^2\|y\|_2^2}$ |
| **Geodesics Distance** | $d_\kappa(x,y) = 2 \tan_\kappa^{-1}(\|-x \oplus_\kappa y\|_2)$ |

### 3.2 $\tan_\kappa^{-1}(x)$ 定义 (MCKG 论文 §3.1 补全)

| κ | 公式 |
|---|------|
| κ > 0 | $\kappa^{-1/2} \tan^{-1}(x \kappa^{1/2})$ |
| κ = 0 | $x - \frac{1}{3}\kappa x^3$ (Taylor 极限) |
| κ < 0 | $|\kappa|^{-1/2} \tanh^{-1}(x \|\kappa\|^{1/2})$ |

### 3.3 真统一性质

| 性质 | R137 abs-based | 新 tan_κ-based |
|------|----------------|----------------|
| κ=0 处梯度 | ❌ `.abs()` 不可导, 梯度=0 | ✅ Taylor 极限 $x - \frac{1}{3}\kappa x^3$ 连续可导, 梯度=$-x^3/3$ |
| 公式分支 | 1 (但 `.abs()` 等价分支) | 1 距离公式 + 3 tan_κ 分支 (每个 branch 内部连续) |
| 连续性 | C⁰ 不连续 at κ=0 | C¹ 连续 at κ=0 |
| Poincaré 兼容 (κ<0) | ✅ | ✅ |
| 球面兼容 (κ>0) | ✅ | ✅ |
| 欧氏兼容 (κ=0) | ✅ | ✅ (Taylor 极限) |

---

## 4. 实施路径

### Phase 0 (立即, 0 GPU): 数学实现 + 数值正确性验证

1. ✅ 创建本 description (R9 max+1 = 333)
2. ⏭️ 写 Python 实现 `scripts/task333_issue42_unified_k_stereographic_formula.py`:
   - `mobius_addition(x, y, kappa)`: 单条 closed-form 加法 (Table 1)
   - `tan_kappa_inverse(x, kappa)`: 3 分支 tan_κ⁻¹ (Taylor 在 κ=0 极限连续)
   - `unified_k_stereographic_distance(x, y, kappa)`: 完整 Table 1 距离公式
3. ⏭️ 数值正确性测试:
   - κ=0: d² = ‖x-y‖² (跟 Euclidean 完全一致, 数值误差 < 1e-5)
   - κ=-1: d² = Poincaré ball distance with c=1 (跟 `hgvae.manifolds.PoincareBall.distance` 一致)
   - κ=+1: d² = 球面距离 (跟 spherical geodesic distance 一致)
4. ⏭️ 梯度可导性测试:
   - κ=0 处 d/dκ ≠ 0 (vs R137 d/dκ = 0)
   - autograd 通过, 反向传播不报错

### Phase 1 (可选, 0 GPU): 对称初始化实验设计 (不启动 GPU)

按 Issue #42 owner 建议:
- θ_init = +0.01 (现状)
- θ_init = -0.01 (双曲起步)
- θ_init = ±0.01 随机符号
- θ_init = N(0, σ²) Gaussian

**仅设计**, 不启动 GPU 训练 (当前 free-curv 主线 NO-GO, R10 backlog 没有重启动机).

---

## 5. 决策记录 (R11.5 透明决策)

| 选项 | 收益 | 成本 | R11.5 决策 |
|------|------|------|-----------|
| 实现 MCKG Table 1 统一公式 | 高 (根治 θ=0 死点) | 中 (数学实现 + 数值验证 ~3h CPU) | ✅ **YES** (本任务) |
| 立即重启 free-curv 训练 | 高 (验证统一公式是否带来 R@10 增益) | 高 (≥1 周 GPU 实验) | ❌ NO (跟当前 NO-GO 闭环一致) |
| 仅 record 不实现 | 低 | 零 | ❌ NO (owner 明确要求"升级为真正的统一公式", 不是仅记录) |
| 实现 + 立即重启 free-curv | 最高 | 高 + 高 (3 GPU days) | ❌ NO (Issue #42 owner "若未来有理由重启"前提未满足) |

**选了**: 实现统一公式 + 数值验证 (Phase 0 + Phase 1 design), 不启动 GPU 训练
**为什么**: 用户 2026-07-30 14:34 明确说 "先做#42的这个issue" → 落实 Issue #42 owner 建议. 但 free-curv 主线 NO-GO (Task #138), 没人 trigger 重启训练 (R10 backlog 都是曲率相关新方向, 跟 free-curv 路线不同), 不应单方面启动 GPU 实验.
**备选**: 立即重启 free-curv 训练 (R11.4 critical decision → 必须 dry-run 先报告再执行, 当前不做)

---

## 6. 跟现有任务兼容性 (R7)

- ✅ 0 GPU work (纯数学 + CPU 数值验证)
- ✅ 不影响 task194 Stage 3 (PID 1986893 on GPU 0)
- ✅ 不影响 task328 (已经 KILLED)

---

## 7. 关键文件 (待写)

| 路径 | 内容 |
|------|------|
| `scripts/task333_issue42_unified_k_stereographic_formula.py` | MCKG Table 1 统一公式实现 + 数值正确性验证 |
| `verdicts/task333_issue42_unified_formula_result.md` | 落地 verdict (数学层 PASS, 不重启 GPU) |
| `HG-Rec/model/hrqvae_free_curv_unified.py` (可选) | 新模块, 替代 R137 (若未来重启 free-curv) |

---

## 8. 关联

- Issue #42 (主)
- verdicts/task331_issue42_free_curv_postmortem_record.md (Issue #42 RECORDED, 本任务是落地)
- HG-Rec/model/hrqvae_free_curv.py:45-76 (R137 实际 abs-based 实现)
- papers/MCKG.md (Table 1 来源)
- Task #137 (R137 自承 θ=0 fixed point)
- Task #89/#138 (free-curv 主线 NO-GO, 不重启)
- Task #142/#144/#145 (κ-decouple 新方向, 若重启需先升级本任务)

---

result: Task #333 Issue #42 κ-Stereographic 统一公式实现 — 设计完成. Phase 0: 实现 MCKG Table 1 距离公式 $d_\kappa(x,y) = 2 \tan_\kappa^{-1}(\|-x \oplus_\kappa y\|_2)$ + tan_κ⁻¹ Taylor 极限在 κ=0 连续 + 数值正确性验证. Phase 1: 对称初始化实验设计 (仅设计不启动). 0 GPU 工作 (R7 兼容), 落实 Issue #42 owner 建议"升级为真正的 κ-stereographic 统一公式", 不重启 free-curv 主线 (跟 Task #138 NO-GO 一致).