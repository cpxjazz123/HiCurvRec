# Issue #95 单 ckpt + beam=20 Verdict (2026-08-09)

## 状态: NO-GO (约束下物理不可达 0.108)

## 核心结果
- test_R@10 = **0.1057** (13 组变体全部此值, 无方差)
- 单 ckpt + beam=20 物理天花板
- 与 0.108 目标差 0.0023

## 4 Gate
- Gate 1 (实施): PASS
- Gate 2 (无崩溃): PASS
- Gate 3 (参数敏感性): FAIL — 零方差
- Gate 4 (vs 0.108): FAIL — 0.1057

## 根因
1. HAB α sigmoid(-20) 已饱和 → 残差贡献 ≈ 0
2. length_penalty 对 max_length=5 固定输出无效
3. 单 ckpt valid/test ratio 1.222 硬过拟合

## 历史对比
| Issue | 方法 | R@10 |
|-------|------|------|
| #141 | 单 ckpt ep130 + beam=30 | 0.1065 |
| #94 | 3-way Borda ensemble | 0.1079 |
| **#95** | **单 ckpt + beam=20 (用户约束)** | **0.1057** |

## 结论
0.108 在单 ckpt + beam=20 框架下本环境不可达。Issue #94 (3-way ensemble 0.1079) 是当前最佳但已被用户约束排除。

## 推荐
接受 0.1057 闭环, 不再尝试单 ckpt + beam=20 框架内的微调。
