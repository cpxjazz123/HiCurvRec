# Task #198 — Stage 3: 放开逐层可学习 κ (c_max·sigmoid(θ) 参数化)

> **任务目的**: 在 Stage 2 双码本解耦成功的基础上, 把固定曲率 c=1.0 改成每层可学习 θ, 通过 c_max·sigmoid(θ) 参数化达到"逐层自适应曲率" — 这是原始 idea 的正面验证.

> **完成日期**: (待 Stage 2 #197 通过后启动)
> **状态**: 🟡 待启动 (设计完成, 排队等 Stage 2 verdict)

---

## 1. 背景

承接 Stage 2 (#197): 双码本 + 钉住 r_target 已激活几何 (λ 进窗口). 现在放开曲率, 让数据自己决定每层最合适的曲率.

**为什么之前 c·tanh(θ) 不可行** (Task #89 free_curv 历史):
| | 之前 c·tanh(θ), κ_max=2 | 现在 c_max·sigmoid(θ), c_max=8.6–25 |
|---|---|---|
| 可达 λ 范围 | [1.76, 2.32] (全在平坦区) | [2, 10⁴] (覆盖整个有用区间) |
| ∂λ/∂c | 0.008 – 0.16 | **6.6 – 112** |
| 边界风险 | κ 涨 → 球壁压到码字上 | **结构性不可能** (‖e‖ 钉死, 球内范数永远 < 1/√c) |
| 可辨识性 | ‖z‖ 可补偿 → 抵消 | **‖z‖ 被钉死 → 唯一旋钮** |

---

## 2. 实验设计

**核心代码**:
```python
# 每层一个可学习曲率
self.theta = nn.Parameter(torch.zeros(1))
c_max = (5.0 / self.r_target) ** 2      # 数值安全上界
# L0: c_max = 25
# L1: c_max = 13.7
# L2: c_max = 8.6
self.c = c_max * torch.sigmoid(self.theta)   # 初始 sigmoid(0)=0.5 → c = c_max/2
```

**必须记录**: `theta`, `c`, `lambda_p50` 三者的逐 epoch 轨迹.

---

## 3. 决策触发

| 结果 | 结论 |
|------|------|
| c 收敛到内部某个值, 且三层不同 | ✅ **可学习曲率首次有效, 而且是逐层自适应的** — 原始 idea 成立 |
| c 收敛但三层相同 | 🟡 曲率有用但不需要逐层 |
| c 仍然乱漂 / 贴边界 | ❌ 还有别的问题, 回到 Stage 2 (#197) 查 |

---

## 4. 多种子验证 (R137 fix 经验)

避免单 seed 噪声, 至少 3 种子 × 3 层 = 9 组 (theta, c, lambda) 轨迹. 报告 mean ± std.

---

## 5. 预算

| 阶段 | 估算时间 |
|------|---------|
| HVectorQuantization 加 θ 参数 (rewrite utils.py:179+) | ~2 h (代码 + 调试) |
| 3 种子 × 500 epoch × 4 GPU 并发 | ~12 h |
| θ/c/λ 三联监控 + 轨迹图 | ~30 min |
| Stage 2/3/4 流水线 | ~80 min |
| verdict | ~30 min |
| 总计 | ~16 h |

---

## 6. 风险与缓解

**风险 1**: θ 漂到边界 (sigmoid 输出 0 或 1) → 缓解: c_max 限制 + 监控 c 轨迹
**风险 2**: 三层 c 都收敛到相同值 → 缓解: init 不同 θ 让三层起始不同 (e.g., [-0.5, 0, 0.5])
**风险 3**: 多种子方差大 → 缓解: 多跑几个种子, 报告方差而非单点
**风险 4**: θ 优化停滞 → 缓解: 给 θ 单独用 Adam(lr=1e-3), 比 codebook lr 大

---

## 7. 完成度跟踪

- [ ] HVectorQuantization 加 θ 参数
- [ ] c_max 计算 (基于 r_target, 三层不同)
- [ ] θ 单独优化器 (lr 大于 codebook)
- [ ] 3 种子 × 3 层 θ/c/λ 三联监控
- [ ] Stage 2 流水线 (Sinkhorn + T5-mini + R@10)
- [ ] verdict 写盘 (verdicts/task198_stage3_per_layer_kappa_result.md)

---

## 8. 与已有任务的关系

| Task | 关系 |
|------|------|
| #89 free_curv product manifold | 历史失败 (κ_max=2 太窄), Stage 3 用 c_max=8.6-25 修复 |
| #195 Stage 0 | 提供 ρ_target 实测基线 |
| #196 Stage 1 | 软约束, Stage 2 前置 |
| #197 Stage 2 | 双码本解耦, Stage 3 前置 |

---

## 9. 后续动作

- **GO (c 收敛 + 三层不同)**: 写论文 — 这是原始 idea 的首次正面验证
- **PARTIAL (c 收敛但三层相同)**: 报告"曲率有用但无需逐层", 简化参数化
- **NO-GO (c 乱漂)**: 回到 Stage 2 查, 或放弃可学习 κ 路线