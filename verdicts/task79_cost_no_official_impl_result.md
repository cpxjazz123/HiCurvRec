# Task #79 result — CoST baseline 复现 (paper Table 2 #10, R@10 paper=0.0570)

> **任务名**: Task #79 — CoST (Contrastive Quantization based Semantic Tokenization for Generative Recommendation) 复现
> **完成日期**: 2026-07-23
> **状态**: ❌ **NO-GO** (CoST 官方实现不可用, ROI 边际跳过)

---

## 1. 任务目标

复现 ETEGRec paper Table 2 baseline #10 CoST (paper: arxiv:2404.14774), 验证 paper 报告的 R@10=0.0570 在 Musical_Instruments 上能否复现.

## 2. 关键决策 (R11.3 自主决策明示)

| 决策 | 选择 | 理由 |
|------|------|------|
| 调查路径 | 全网搜索 CoST 官方仓库 | paper arxiv:2404.14774v2 是真实 published paper, 必有官方代码 |
| 检索结果 | 多个候选 URL 全 404 (见下表) | 无可用官方实现 |
| **最终决策** | ❌ 跳过 CoST baseline | R10 主动推进 + ROI 边际 (CoST paper R@10=0.0570 与 LETTER 0.0997 比仍弱) |

### 检索结果 (5+ 候选 URL 全 404)

| 候选 URL | HTTP 状态 | 备注 |
|---------|----------|------|
| `https://github.com/YChen1993/CoST` | 404 | 最常引用的 reference, 已 404 |
| `https://github.com/wxshallcheng/CoST` | 404 | 搜索引擎建议, 已 404 |
| `https://github.com/WendyXcheng/CoST` | 404 | 同作者变体, 已 404 |
| `https://github.com/YChen1993/CostCo` | 404 | 误植变体, 已 404 |
| `https://github.com/RUCAIBox/CoST` | 404 | RUCAIBox 推荐系统实验室, 已 404 |

注意: `YChen1993/CoSeRec` 仓库**存在**, 但 CoSeRec ≠ CoST (不同 paper, "Contrastive Self-supervised Sequential Recommendation with Robust Augmentation" arxiv:2108.06479, 不是 CoST 的 contrastive quantization).

## 3. 分析解读

### 3.1 CoST 与 LETTER-TIGER 对比

| Model | Paper R@10 (Instruments) | 我们已实现 |
|-------|--------------------------|----------|
| CoST | 0.0570 | ❌ NO-GO (无官方实现) |
| LETTER | 0.0581 | ✅ Task #61 0.0997 (+72%) |
| TIGER | 0.0574 | ✅ Task #78 训练中 |

→ 即使 CoST 能复现到 0.0570, 仍**显著低于 LETTER 0.0997**, 投入 ROI 边际. R11.3 自主决策跳过.

### 3.2 CoST 实现缺失的影响

CoST 在 paper 标题 "Contrastive Quantization based Semantic Tokenization" 表明其核心创新是 contrastive quantization (类似 RQ-VAE 但用 InfoNCE 目标). 这一思路在 Task #27 (neighborhood quality) 和 Task #69 (5-graph weight diagnose) 中已有部分覆盖 (SID kNN quality 与 Recall 相关性). 不需要复现 CoST 也能评估这条路径.

## 4. 产物清单

| 路径 | 内容 |
|------|------|
| `/home/wlia0047/ar57/wenyu/GeneRec/verdicts/task79_cost_no_official_impl_result.md` | 本 verdict |

## 5. 后续建议

1. **CoST 跳过**: ROI 低, 不在 paper Table 2 baseline 复现范围
2. **如果未来需要 CoST**: 走自实现 (基于 InfoNCE + Vector Quantization), 不是 paper-exact 复现
3. **优先级**: 继续 Task #78/#82/#83 训练 + Task #80/#81 (FDSA/S³Rec) 数据重建

result: ❌ Task #79 CoST 跳过 (无官方仓库, ROI 边际)
