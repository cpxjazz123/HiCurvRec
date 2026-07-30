# Task #331 / Issue #42 — Free-Curv 复盘 (κ-stereographic 统一公式缺口) — 📝 RECORDING VERDICT

**日期**: 2026-07-30 14:30
**触发**: Issue #42 owner 2026-07-30 11:55 创建
**状态**: 📝 **RECORDING VERDICT (no code, no training, no GPU)**
**类型**: Post-mortem 文档 (per Issue #42 明确要求 "建议（仅记录，不建议立即执行）")

---

## 1. Issue #42 核心主张 (完整记录)

### 1.1 观察

Task #135 诊断出 `hrqvae_free_curv.py` 4 处硬分支 bug:
- `if k_m.item() == 0.0` 分支 (κ=0 走欧氏分支)
- `.item()` 把 κ 从计算图 detach (κ<0 / κ>0 走 torch.where 但分支 detach)

### 1.2 推荐修复方案 A (Sala/MCKG Table 1 风格)

用 `tan_κ / tanh_κ` 类连续函数, 把球面/双曲/欧氏三种情况用同一条数学公式表达:
- κ 在整个定义域 (含 κ=0) 连续可导
- 理论上不存在 "选中哪一支公式" 这个概念

### 1.3 实际 R137 落地代码 (3-branch + torch.where)

```python
eucl_sq = ((x - c) ** 2).sum(dim=-1)                    # 完全不含 κ
sph_sq  = (torch.acos(cos_theta) / sqrt_kappa) ** 2      # 球面分支
hyp_sq  = poincare_distance(x_h, c_h, c_tensor) ** 2     # 双曲分支
out = torch.where(k_m > 0, sph_sq, torch.where(k_m < 0, hyp_sq, eucl_sq))
```

仍然是 **3 套独立公式**, 只是把 Python `if` 换成 `torch.where`, **不是真正的统一公式**.

### 1.4 直接后果

1. **eucl_sq 分支不依赖 κ**: 即使 R137 修复完全生效, θ=0 init 仍是数学 fixed point
2. **Task #137 verdict 自承**: "θ=0 init 仍是数学 fixed point (eucl branch 数学上不依赖 κ, torch.where 的反向传播只走 selected 分支)"
3. **现有工程绕开**: θ_init=0.01 而非 0 (人为选正数, 训练结果天然偏向球面分支)
4. **从未真正验证过双曲侧起步**: 起始点符号 θ=+0.01 人为决定了最终学到球面 vs 双曲, 存在 **混杂因素**

---

## 2. 跟当前任务状态的关系

| 任务 | 状态 | 跟 Issue #42 关系 |
|------|------|-------------------|
| Task #89 (free-curv 原始) | ❌ ABANDONED | 主线 NO-GO |
| Task #135 (4 处硬分支 bug) | ✅ DONE | Issue #42 直接引用 |
| Task #137 (R137 修复) | ✅ DONE | Issue #42 复盘核心 |
| Task #138 (geodesic kmeans + dead-code reset) | ❌ NO-GO | 决定性诊断 codebook 坍缩 |
| Task #142 (#137 κ-decouple Stage 3) | ❌ NO-GO | 后续尝试也失败 |
| Task #144 (κ+codebook 解耦 Stage 2 only) | ⏭️ OPEN backlog | 若重启需考虑 Issue #42 修复 |
| Task #145 (软量化退火) | ⏭️ OPEN backlog | 同上 |

---

## 3. R11.5 透明建议 (仅记录, 不立即执行)

按 Issue #42 明确要求 "若未来有理由重启 free-curv 方向..." 建议记录:

### 3.1 优先级 1: 真正的 κ-stereographic 统一公式

替代当前 3-branch+torch.where:

```python
# 候选统一公式 (Sala 2018 / MCKG Table 1 风格)
def unified_k_stereographic_distance(x, c, kappa, d):
    """单一连续公式, κ=0 极限回到欧氏, κ>0 球面, κ<0 双曲"""
    # 反演 u = ((1 - ||c||²)·x + (1 + κ·||x||²)·c) / ...  (具体需核实)
    # 距离 d(u_x, u_c) 在 κ=0 处取极限 = arccosh
    ...
```

**实施前置**:
- 需核实 MCKG Table 1 公式是否真有连续 κ→0 极限
- 需验证反向传播在 κ=0 邻域稳定
- 需做 numerical gradient check vs analytical

### 3.2 优先级 2: 对称初始化实验

替代当前固定 θ_init=+0.01:
- θ_init = +0.01 (现状)
- θ_init = -0.01 (双曲起步)
- θ_init = ±0.01 随机符号 (对照组)
- θ_init = N(0, σ²) (Gaussian 初始化)

**目的**: 排除 "起始点符号人为决定最终学到球面 vs 双曲" 混杂因素

### 3.3 决策门

**仅当以下条件全部满足才考虑重启**:
1. Task #144 / #145 等 κ-decouple 新方向被证明可行 (打破 codebook 坍缩)
2. 跟 Issue #30 per-layer Codebook Transforms 协同能稳定 hit NORTH STAR R@10 > 0.1053
3. 有明确 evidence "当前 3-branch 写法确实限制性能" (而不是预先空想)

---

## 4. 决策记录 (R10 推进 + R11.5 自主决策)

### 4.1 决定

**Issue #42 关闭 (RECORDED, NO-GO)**:
- 不重写统一 κ-stereographic 公式 (现在没有重启动机)
- 不做对称初始化实验 (同上)
- 仅将 Issue #42 内容登记到本 verdict (本文件)
- GitHub Issue #42 评论: 登记 verdict 路径 + R11.5 透明记录

### 4.2 R11.5 决策矩阵

| 选项 | 收益 | 成本 | R11.5 决策 |
|------|------|------|-----------|
| 立即重写统一公式 | 高 (根治 θ=0 死点) | 高 (GPU 训练 + 验证) | ❌ NO (没有重启动机) |
| 仅记录 verdict | 低 (合规 Issue #42) | 低 (零 GPU) | ✅ YES (本任务) |
| 完全 ignore Issue #42 | 零 | 零 (破 R14) | ❌ NO (R14 强制处理) |

### 4.3 R11.5 透明选择

**选了**: 仅记录 verdict (本任务)
**为什么**: Issue #42 明确要求"仅记录, 不建议立即执行". Free-curv 主线 NO-GO (Task #138), 当前 backlog 都是新方向 (#30/#34/#41). 没有人 trigger 重启统一公式, 单方面做会浪费资源.
**备选**: 立即重写统一公式 + 对称初始化 (需要 GPU + 至少 1 周实验, 不在 R14 闭环范围)

---

## 5. 关键文件 + 物理产物

| 文件 | 用途 |
|------|------|
| `verdicts/task331_issue42_free_curv_postmortem_record.md` | 本文件 (RECORDING verdict) |
| Issue #42 GitHub 评论 | 登记 verdict 路径 + R11.5 透明决策 |

---

## 6. 关联引用

- Issue #42 (主, post-mortem)
- Task #89 (free-curv 原始, ABANDONED)
- Task #135 (4 处硬分支 bug 诊断)
- Task #137 (R137 修复 verdict, 自承 θ=0 是 mathematical fixed point)
- Task #138 (geodesic kmeans + dead-code reset, NO-GO 终审)
- Task #142 / #144 / #145 (κ-decouple 新方向, 若重启需考虑 Issue #42 修复)
- memory `free-curv-codebook-collapse.md` (codebook 坍缩架构根本问题)

---

## 7. R14 闭环

- Issue #42 RECORDED verdict 已写 (本文件)
- GitHub Issue #42 评论: 登记 verdict 路径 + R11.5 透明决策
- 跟 R10 推进 + R11.5 自主决策一致: 主动记录但不越权启动重启

---

result: Issue #42 Free-Curv 复盘 RECORDED (📝 no-code verdict). Task #137 R137 的 3-branch+torch.where 不是真正 κ-stereographic 统一公式, θ=0 死点未根解 (Task #137 verdict 自承). 仅记录不重启, 因 free-curv 主线 NO-GO (Task #138) + 当前 backlog 全新方向 (Issue #30/#34/#41) 没有重启动机. R11.5 透明决策: 选 "仅记录", 备选 "立即重写统一公式" 因 GPU 成本 + 缺动机被拒.
