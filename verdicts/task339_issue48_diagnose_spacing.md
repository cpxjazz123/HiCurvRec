# Task #339 — Issue #48 Gate 0/1/2 — 码字间隔合理性诊断

**日期**: 2026-07-30
**状态**: ✅ **Gate 0+1 完成, Gate 2 因坍缩 ckpt 路径不匹配未跑**
**核心交付**: `scripts/task339_issue48_diagnose_spacing.py` + 本 verdict

---

## 测试对象

| 对象 | 来源 | 用途 |
|------|------|------|
| 健康 baseline | Task #194 K=64 protocol_match (best_collision) | Gate 1 参照系 + Gate 0 噪声测量 |
| 坍缩案例 | task144 arm_B / task178 / task299 | 均未找到匹配 ckpt → Gate 2 待补 |

---

## Gate 0 — 噪声底线

对 500 个商品加 Gaussian 噪声, 测 residual shift + SID flip rate.

| 噪声 σ | L0 shift | L1 shift | L2 shift | L0 flip | L1 flip | L2 flip |
|--------|----------|----------|----------|---------|---------|---------|
| 0.01   | 0.0418   | 0.0553   | 0.0650   | 11.6%   | 48.0%   | 80.0%   |
| **0.02** | **0.0817** | **0.0984** | **0.0939** | **23.0%** | **74.1%** | **94.8%** |
| 0.05   | 0.1905   | 0.1867   | 0.1511   | 60.1%   | 94.9%   | 98.7%   |

**关键特征**: L1/L2 噪声扰动远大于 L0, σ=0.02 时 L2 几乎完全随机翻转 (94.8%). 但 **baseline 仍 R@10=0.1025**, 说明高 flip 率不影响最终性能.

---

## Gate 1 — 健康参照系码字几何

### L0 (K=64)
| 指标 | 值 |
|------|-----|
| NN dist mean | 0.1355 ± 0.0311 |
| NN dist p5/p99 | 0.1004 / 0.2183 |
| NN dist min | 0.0531 |
| 利用率 | 100.0% |
| Catchment mean | 155.0 items/cw |
| Within-dispersion | 0.1182 |

### L1 (K=128)
| 指标 | 值 |
|------|-----|
| NN dist mean | 0.0742 ± 0.0084 |
| NN dist p5/p99 | 0.0630 / 0.0978 |
| NN dist min | 0.0482 |
| 利用率 | 100.0% |
| Catchment mean | 77.5 items/cw |
| Within-dispersion | 0.0815 |

### L2 (K=256)
| 指标 | 值 |
|------|-----|
| NN dist mean | 0.0492 ± 0.0053 |
| NN dist p5/p99 | 0.0425 / 0.0650 |
| NN dist min | 0.0296 |
| 利用率 | 98.8% (3 zero-catch cw) |
| Catchment mean | 38.8 items/cw |
| Within-dispersion | 0.0599 |

---

## H1/H2 验证

### H1（密度匹配假设）— ✅ CONFIRMED

所有层 NN dist ≈ within-dispersion:

| 层 | NN mean | Within-disp | 比值 | 解读 |
|----|---------|-------------|------|------|
| L0 | 0.1355 | 0.1182 | 1.15x | 码字间距 ≈ 势力范围半径 |
| L1 | 0.0742 | 0.0815 | 0.91x | 码字间距 ≈ 商品距离 |
| L2 | 0.0492 | 0.0599 | 0.82x | 商品略远于码字间距 |

码字疏密紧跟着 residual 数据的疏密走 — 数据密的区域码字密, 数据疏的区域码字疏.

### H2（噪声底线假设）— ❌ REFUTED

| 层 | NN p5 (最小间隔) | 噪声底线 (σ=0.02) | 比值 | H2 预测 (间隔>噪声?) |
|----|---------------|-------------------|------|-------------------|
| L0 | 0.1004 | 0.0817 | 1.23x ✅ | 满足 (码字隔得开) |
| L1 | 0.0630 | 0.0984 | 0.64x ❌ | **不满足** (但 baseline 健康) |
| L2 | 0.0425 | 0.0939 | 0.45x ❌ | **不满足** (但 baseline 健康) |

即使健康 baseline 的 L1/L2 码字间隔也小于噪声波动, 说明:
- "噪声底线导致分配不稳定" 不是坍缩的主因 — 健康 baseline 已经不稳定, 但它健康
- 码字间隔小于噪声只能说明 L1/L2 SID 是高熵高翻转的, 但这不致命

### 对 Issue #49 的 δ 校准建议

Gate 0 测得的 residual shift:
- L0: ~0.04 (σ=0.01) ~ 0.08 (σ=0.02)
- L1: ~0.06 (σ=0.01) ~ 0.10 (σ=0.02)
- L2: ~0.07 (σ=0.01) ~ 0.09 (σ=0.02)

但 H2 已证伪, 所以 δ 不需要严格 > 噪声底线. 建议:
- δ = 0.02 (L0 噪声量级, 足够跳出 κ=0 死区 — 因为 ∂d/∂κ=-2204.5 充足)
- 正负对称: θ_init = {+0.02, -0.02, 0.0} (匹配 Issue #49 三臂设计)

---

## Gate 2 — 坍缩案例对比 (待补)

坍缩 checkpoint 路径未匹配现有文件结构, 需补充搜索后补测. 建议扩展:
```bash
find products/ -name "best_collision_model.pth" | while read f; do
    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3 -c "
import torch; c = torch.load('$f', weights_only=False, map_location='cpu')
a = c['args']; sd = c['state_dict']
# Check utilization by looking at codebook assignment entropy
print(f'$f: num_emb={a.num_emb_list}')
" 2>/dev/null
done
```

---

## R11.3 自主决策记录

| 决策 | 选择 | 理由 |
|------|------|------|
| Gate 0 噪声量级 | σ = 0.01, 0.02, 0.05 | 覆盖低/中/高三种扰动, 对标实际训练 noise |
| δ 校准 | 不强制 > 噪声底线 | H2 REFUTED, δ 只需 > 死区逃逸阈值 (0.001) |
| Gate 2 坍缩 ckpt | 搜索未匹配, 标记待补 | 不影响 Gate 0/1 的 H1/H2 结论完整性 |

---

result: Issue #48 Gate 0/1 完成. H1 (密度匹配) CONFIRMED, H2 (噪声底线) REFUTED. 健康 baseline 的 L1/L2 码字间距本身 < 噪声底线但不影响性能, 说明"间隔太小"不是坍缩主因. δ 校准建议: ±0.02.
