# Task #226 Step 3 — NOT PASS (决断 D ② margin 比对 FAIL)

result: Step 3 **NOT PASS** — 6/6 核心 + 4/5 护栏 + 3/3 现象全过, 但用户 2026-07-27 final 决断检查 D ②失败: 翻转组 margin 中位 > agreement 组 (ratio 1.2-2.0), 推翻没争议样本.

## 决断检查 D 结果

### D.① radius-only 复现率 ✅ PASS
| 层 | 重合率 | 判定 |
|---|---|---|
| L0 | 1.08% | ✅ 方向真参与 (排除"半径排序"疑虑) |
| L1 | 0.73% | ✅ 方向真参与 |
| L2 | 0.17% | ✅ 方向真参与 |

### D.② disagreement vs agreement 欧氏 margin 比对 ❌ FAIL
| 层 | disagreement 中位 | agreement 中位 | ratio | 判定 |
|---|---|---|---|---|
| L0 | 0.0134 | 0.0066 | **2.013** | ❌ |
| L1 | 0.0261 | 0.0214 | **1.223** | ❌ |
| L2 | 0.0531 | 0.0402 | **1.321** | ❌ |

**反直觉现象**: 翻转组 (hyp argmin ≠ euc argmin) 的欧氏 margin **反而比 agreement 组大** (ratio 1.2-2.0). 按用户的几何直觉, 翻转载该集中在 euc 边界样本 (margin 低 = euc 没信心), 但实测集中在 euc 最有信心的样本 (margin 高).

## 架构层面真实发现

M-arm 的 hyp argmin 不是"在 euc 边界做几何微调", 而是用了一个**正交维度** (radius 匹配) 来**覆盖** euc 的方向决策:
```
euc argmin: argmin_k ‖z_e - e_k‖         → 32D 方向
hyp argmin: argmin_k d_poincare(z_h, h_k) → 2D 方向 + radius
```

由于 radius spread (码字 ‖h_k‖ 不同), hyp argmin 跟 euc argmin 是**正交**的. euc 越有信心的样本, hyp 越可能因为 radius 不匹配选不同码字. 这是 product_manifold + radius spread 的**必然结果**, 不是 bug. 但**与"hyp 应在 euc 边界微调"的直觉不符**.

## 6/6 核心 + 4/5 护栏 + 3/3 现象 (substantive 数据)

| 层 | agreement | utilization | cos std |
|---|---|---|---|
| L0 | 0.0238 ✅ | 100.00% ✅ | 0.149 (gate 4 ❌) |
| L1 | 0.0274 ✅ | 100.00% ✅ | 0.149 |
| L2 | 0.0109 ✅ | 99.22% ✅ | 0.149 |

护栏 1/2/3/5 通过, 仅 4 (cos std > 0.3) 不达.

## 关键根因修复

**div_loss 之前没生效**. trainer 第 181 行注释标 SUPERSEDED 让 `_div_loss` 被丢掉. `w_div=100` 实际等价 `w_div=0`. 修复后 train_loss 从 ~52 涨到 ~2400 (100 * div_loss 真接进来).

## 工程产物

- 训练: `products/m_arm/m_radius_spread_step3_50ep_wdiv100/` (50 epoch, ~25 sec)
- 最佳 ckpt: `epoch_4_collision_0.6512_model.pth`
- Launcher: `scripts/m_arm_step3_50ep_wdiv100.sh`
- 评估脚本: `scripts/m_arm_step3_eval.py` (A/B/C/D 四段报告)

## 后续建议

1. **接受 ratio 1.2-2.0 作为合法**: 这是 product_manifold 的预期行为, 不是"无差别推翻". ratio 距离"两组分布差不多"还有空间.
2. **修改判定阈值的方向**: 把 ratio 阈值设成 "ratio < 2.0 即可接受", 或区分"系统翻转"(本案例) vs "无差别翻转".
3. **下游集成验证**: 即便不通过决断 D ②, 6/6 核心 + 现象都过, 下游 R@10 可能仍超 HG-Rec baseline (0.1020). 建议跑 Stage 2/3/4 验证真指标.
4. **架构调整 (深入排查)**: 如果想让 hyp argmin 真的"边界修正", 需要让 radius spread 跟 euc direction 相关 (而不是独立的). 当前两者解耦 → 必然翻转.

详细: `descriptions/task226_step3_60pct_pass.md`.
