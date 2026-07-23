# Task #90 — Phonism vs HG-Rec Codebook 机制分解 (analytical, no GPU training)

> **任务目的**: 用现有 RQ-VAE ckpt 对比 vanilla RQ-VAE + Sinkhorn (phonism R@10=0.1058) vs Hyperbolic κ=0.5 (HG-Rec c555 R@10=0.1051) 的 codebook 几何 + SID 分布差异, 解释为何 c555 微弱最优但 free-curv 训练后 κ→0 (Task #89)

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

> **说明**: 本任务编号 90 与 `verdicts/task90_result.md` (FMLP-Rec NO-GO) 是不同 task — verdicts 命名空间与 descriptions 命名空间独立 (R9 仅管 descriptions), 历史 verdict 保留.

---

## 1. 背景

### 1.1 表面矛盾

- **Task #88 c555 微弱最优**: 6 网格 R@10=0.1051, 比 c111 高 +5.3%
- **Task #89 free-curv → κ=0**: 18/18 (layer, κ_m) 训练后 = 0.000000
- **Task #84 baseline R@10=0.1020**: HG-Rec c111 (κ=1.0) 标准 baseline

**矛盾**: 如果数据本质欧氏 (Task #89), 为什么预设 κ=0.5 (c555) 比 κ=1.0 (c111) +5.3%? 但比 κ=0 (free-curv A 臂下游 R@10=0.1015) 又高 +3.6%?

### 1.2 假设

- **H1**: c555 的 +5.3% 不是几何效应, 是 codebook 初始化半径 (K-Means 初始化的隐式 spread) 与 κ=0.5 配合产生的伪几何优势
- **H2**: c555 的 codebook 实际训练后 κ→0 但 K-Means 初始化阶段的 spread 被保留 → 训练过程 = "用欧氏梯度退火 hyperbolic init"
- **H3**: c555 是 overfitting noise (单 seed, 5.3% 在 ±5% 噪声区间)

### 1.3 复用现有产物

| 来源 | ckpt | 描述 |
|------|------|------|
| Task #84 | `products/task84/.../best_loss_model.pth` | HG-Rec c111 (κ=1.0) |
| Task #88 | `products/task88/train/curv_0.5_0.5_0.5/best_loss_model.pth` | HG-Rec c555 (κ=0.5) |
| Task #89 | `products/task89/train/arm_A_M1/best_loss_model.pth` | HG-Rec free-curv A 臂 (κ→0) |
| Task #32 / phonism | `logs/phonism_*/.../rqvae_inference_flat/` | vanilla RQ-VAE + Sinkhorn |

---

## 2. 实验设计

### 2.1 变量

**比较 4 个 RQ-VAE 配置的 codebook + SID 输出**:
1. **vanilla (Task #32 phonism)**: MSE loss + Sinkhorn post-processing
2. **HG-Rec c111 (Task #84)**: Poincaré loss + κ=1.0
3. **HG-Rec c555 (Task #88)**: Poincaré loss + κ=0.5
4. **HG-Rec free-curv (Task #89 A 臂)**: Poincaré loss + κ→0 自学

### 2.2 保持不变

- **输入**: `HG-Rec/dataset/Instruments/item_emb.parquet` (sentence-t5-base 768d, 9922 items)
- **数据 split**: 完整 9922 items 全量推理 (无 split)
- **seed**: 42 (与 Task #84/88/89 一致)

### 2.3 测量项

**Codebook 几何**:
- 每个 codebook 层 (L0=64, L1=128, L2=256) 的 vector norm 分布
- 每个 codebook 层 pairwise distance 分布 (用对应几何的距离: vanilla Euclidean, c111/c555 Poincaré, free-curv final κ→0 Euclidean)
- Intra-cluster spread (item-to-assigned-codebook-vector 平均距离)
- Codebook 利用率 (per-layer unique codes)

**SID 分布**:
- L0/L1/L2 token 分布 (是否均匀, 是否有 hot tokens)
- 3-token SID 碰撞数 (相同 SID 的 item 数)
- 4-token SID (with dedup) 唯一率
- Per-layer entropy (高熵 = 均匀, 低熵 = 集中)

**下游 T5 训练信号**:
- Stage 3 训练 loss 曲线 (从 `products/task88/.../events.out.tfevents.*` 读)
- Stage 3 验证 R@10 轨迹
- Stage 4 测试 R@5/R@10/NDCG (已闭环)

### 2.4 启动命令

**不需要 GPU 训练, 只需 GPU 加载 ckpt + 推理 1 次**:

```bash
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /home/wlia0047/ar57/wenyu/GeneRec

# 加载 4 个 RQ-VAE ckpt, 对 item_emb.parquet 推理, 计算 codebook 几何 + SID 分布
python3 scripts/task90_codebook_decomposition.py \
    --input_parquet HG-Rec/dataset/Instruments/item_emb.parquet \
    --ckpt_c111 products/task84/.../best_loss_model.pth \
    --ckpt_c555 products/task88/train/curv_0.5_0.5_0.5/best_loss_model.pth \
    --ckpt_free_curv products/task89/train/arm_A_M1/best_loss_model.pth \
    --output_json verdicts/task90_codebook_decomposition.json \
    --output_md verdicts/task90_codebook_decomposition_result.md \
    --gpu 0
```

**预期时间**: 加载 ckpt ~1 min + 推理 9922 items × 4 ckpts ~2 min + 分析 ~1 min. **总计 ~5 min**, 1 GPU.

---

## 3. 决策触发

| 观察 | 决策 |
|------|------|
| **H1 确认**: c555 codebook 初始化半径显著大于 c111 (因 Poincaré 球内 κ=0.5 比 κ=1.0 大), 但训练后收敛半径相近 → 解释 c555 marginal 优势是 init artifact | ✅ 写论文 Section 5.4: "HG-Rec c555 marginal best is a codebook init artifact, not a geometric effect" |
| **H2 部分**: c555 codebook 训练后 κ→0 但 vector norm 分布偏离 vanilla → 仍有非几何差异 | ⚠️ PARTIAL → 报告 "init spread retained even after κ→0 training" |
| **H3 确认**: c555 vs c111 所有 codebook 几何指标在 ±5% 噪声内, 下游 R@10 5.3% 差完全在统计噪声内 | ❌ 否证 c555 真有优势 → 报告 "no meaningful difference between c555 and c111" |
| 全部 H1/H2/H3 否证 | ❌ 矛盾原因不明 → 建议 future work multi-seed 验证 (但用户禁用 multi-seed) |

---

## 4. 预算

| 阶段 | 估算时间 | GPU |
|------|---------|-----|
| 加载 4 个 ckpt + 推理 | ~5 min | GPU 0 |
| Codebook 几何 + SID 分布分析 (numpy/scipy) | ~2 min | 无 |
| 写 verdict + 论文 Section 5 段落 | ~15 min | 无 |
| **总计** | **~25 min** | **~5 GPU-min** |

---

## 5. 风险与缓解

**风险 1**: 4 个 ckpt 路径不一致 (Task #84/#88/#89 路径格式不同) → 写脚本时先 `find products/task{84,88,89}/ -name "best_loss_model.pth"`
**风险 2**: phonism (Task #32) ckpt 路径可能已删除 (loop 2026-07-19 cleanup) → fallback: 直接用 codebook .npy 文件 (`HG-Rec/dataset/Instruments/Instruments_*.npy`)
**风险 3**: HG-Rec free-curv A 臂 ckpt 可能丢失 → 验证存在性, 缺失时跳过 A 臂, 只对比 3 个
**风险 4**: RQ-VAE 模型架构不同时 (HRQ-VAE vs vanilla) → 写统一 wrapper 处理两套架构

---

## 6. 完成度跟踪

- [ ] 写 `scripts/task90_codebook_decomposition.py`
- [ ] 验证 4 个 ckpt 路径存在
- [ ] 加载 ckpt + 推理 (5 min)
- [ ] 计算 codebook 几何指标
- [ ] 计算 SID 分布指标
- [ ] 读 Stage 3 loss 曲线
- [ ] 综合分析 (H1/H2/H3 验证)
- [ ] 写 verdict `verdicts/task90_codebook_decomposition_result.md`
- [ ] 更新 loop.md §16 (R8 归档)
- [ ] 论文 Section 5.4 段落草稿

---

## 7. 关联

- 前置: Task #84 (HG-Rec baseline), Task #88 (per-layer curvature), Task #89 (free-curv)
- 并行: Task #87 v2 synthesis (已闭环)
- 后续: 论文 Section 5 写作 (Section 5.4 Table 2 + Section 5.5 mechanism decomposition)

---

**核心问题**: phonism 0.1058 vs HG-Rec c555 0.1051 vs c111 0.1020 vs free-curv 0.1015 — **同源数据 4 种实现, R@10 跨 4.3% [0.1015, 0.1058]**. 解释这个 4.3% 区间的物理来源.