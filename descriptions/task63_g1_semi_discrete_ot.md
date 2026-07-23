# Task #63 — G1: 半离散最优传输量化 — 码本塌缩的几何根治 (方法主线)

> **任务目的**: 用 Sinkhorn-balanced k-means (semi-discrete OT) 替换 Simple KMeans, 验证 100% 码本利用率是否能在失真税 < 10% 下达到, 以及 G1-H2: 均衡 L1 剖分是否能把信息往深层推 (V-information 剖面去集中化)
> **执行日期**: (待启动, 优先级 2/5)
> **状态**: 🟡 待启动

---

## 1. 背景

承接 Task #58-#61 Campaign (最高 R@5 = 0.0977, Simple KMeans 已稳定为 Stage 2 标准). RQ-VAE / RVQ 的码本塌缩是已记录现象, 现有修法 (EMA restart, 熵正则, 随机重启) 全是启发式补丁.

**几何重构**: 标准 VQ/k-means 等价于**自由质量** Wasserstein 量化, 最优解的胞元是 Voronoi 剖分 (Lloyd 迭代), 质量 $m_k$ 不受约束 → **塌缩就是质量分配的退化解** (大量 $m_k \to 0$).

**改为半离散 OT**: 强制 $m_k = 1/K\ \forall k$, 利用率 = 100% **由构造保证**, 不需要辅助损失. 最优胞元是 **Laguerre / power 胞元** (代替 Voronoi), 分配规则仍是一次 argmin (对修正距离), **推理成本与普通 VQ 完全相同**.

**假设 G1-H1 (D0 GO)**: Sinkhorn-balanced k-means 在失真税 < 10% 下达到接近 100% 利用率.
**假设 G1-H2 (P1)**: 均衡 L1 剖分产生更平稳的残差场, 逐层 V-information 剖面显著去集中化.

---

## 2. 实验设计

**变量**: Stage 2 量化算法 (Simple KMeans → Sinkhorn-balanced k-means)
**保持不变**:
- Stage 1 embedding (复用 Task #59 flan-t5 2048d)
- seed=42, K=256, num_hierarchies=3 (Stage 2) → 4 (Stage 3+4)
- Stage 3 TIGER 配置 (与 Task #59 完全一致)
- Stage 4 评估脚本

**启动命令**:
```bash
# D0: L1 Sinkhorn-balanced k-means vs vanilla k-means (1 天, 纯 GPU KMeans)
python3 scripts/task63_sinkhorn_l1.py \
    --input_pt logs/task59_s1/runs/2026-07-20/04-05-08/pickle/merged_predictions_tensor.pt \
    --output_dir logs/task63_s2_d0/pickle \
    --K 256 --sinkhorn_eps 0.05 --num_iters 50 --seed 42

# P1: 完整 RQ 级联版 (每层 Laguerre 分配) (3 天)
python3 scripts/task63_rq_laguerre.py \
    --input_pt logs/task59_s1/runs/.../merged_predictions_tensor.pt \
    --output_dir logs/task63_s2/pickle \
    --K 256 --num_hierarchies 3 --sinkhorn_eps 0.05 --seed 42

# P2: η 扫描 Pareto 曲线 (2 天, 多个 seed 并行)
for eta in 0.0 0.3 0.5 0.7 0.9; do
    python3 scripts/task63_rq_laguerre.py --eta $eta --output_dir logs/task63_s2_eta${eta}/pickle
done

# P3: 端到端 TIGER (沿用战线四协议)
CUDA_VISIBLE_DEVICES=1 python3 -m src.train experiment=tiger_train_flat \
    data_dir=data/amazon_data/toys \
    semantic_id_path=logs/task63_s2/pickle/cluster_ids.pt \
    sequence_length=120 num_hierarchies=4 task_name=task63_s3_train

CUDA_VISIBLE_DEVICES=1 python3 -m src.inference experiment=tiger_inference_flat \
    data_dir=data/amazon_data/toys \
    semantic_id_path=logs/task63_s2/pickle/cluster_ids.pt \
    ckpt_path=/home/wlia0047/ar57/wenyu/GeneRec/logs/task63_best.ckpt \
    sequence_length=120 num_hierarchies=4 task_name=task63_s4_infer

python3 scripts/task58_recall_eval.py \
    --constrained_pt logs/task63_s4_infer/runs/<date>/<time>/pickle/merged_predictions_tensor.pt \
    --sid logs/task63_s2/pickle/cluster_ids.pt \
    --ckpt /home/wlia0047/ar57/wenyu/GeneRec/logs/task63_best.ckpt \
    --out_json verdicts/task63_recall_eval.json
```

---

## 3. 决策触发 (vs 提案 D0 GO 条件)

| D0 指标 | 决策 |
|---------|------|
| PPL 接近 K (≥ 0.95K) 且 D_rel 劣化 < 10% (vs Simple KMeans) | ✅ GO → P1 完整 RQ 级联 + P2 η 扫描 + P3 TIGER |
| PPL 接近 K 但 D_rel 劣化 10-30% | ⚠️ PARTIAL → 调 sinkhorn_eps (更小 = 更严均衡, 失真税更高), 反复到 < 10% 再进 P1 |
| PPL < 0.5K 或 D_rel 劣化 > 30% | ❌ NO-GO → 算法或实现问题, 切换到 EMA-restart / 熵正则基线 |
| P3 R@5 ≥ 0.0977 (Task #61) | ✅✅ 端到端胜利, 与 Simple KMeans 持平或超越 |
| P3 R@5 < 0.0977 | ❌ 端到端失败, 留作 verdict 反例 (Sinkhorn 平衡约束与下游 Recall 关系) |

---

## 4. 预算

| 阶段 | 估算时间 | GPU |
|------|---------|-----|
| P0 文献核查 (1 周) | 半天 | 无 |
| **D0** L1 Sinkhorn vs vanilla | **1 天** | cuda:0 (~5 min/run) |
| P1 完整 RQ 级联 (每层 Laguerre) | 3 天 | cuda:0 (~15 min/run) |
| P2 η Pareto 扫描 (~5 个 seed) | 2 天 | cuda:0 (~1 hour total, 并行) |
| P3 Stage 3 TIGER | 3 天 | cuda:1 (~120 min, 与 Task #61 一致) |
| P3 Stage 4 + 评估 | 5 min + 1 min | cuda:1 |
| **总计** | **~10 天** | **~10 GPU-hour** |

---

## 5. 风险与缓解

**风险 1**: Sinkhorn-balanced k-means 需新写算法实现 → 参考 POT (Python Optimal Transport) 库的 `ot.sinkhorn` + 自定义 Lloyd 循环
**风险 2**: RQ 级联版每层都需调 sinkhorn_eps → 先在 L1 调稳, 深层继承同一 eps, 必要时 L2/L3 微调
**风险 3**: Laguerre 胞元分配可能比 Voronoi 慢 2-3x → 已预知可接受, Stage 2 ~5s vs ~15s
**风险 4**: G1-H2 验证需要 V-information 测法 (已有 4 种 from 战线一) → P1 阶段同步复用

---

## 6. 完成度跟踪

- [ ] P0 文献核查 (检索词 `balanced vector quantization Sinkhorn`, `semi-discrete optimal transport codebook`, `RQ-VAE codebook collapse optimal transport`)
- [ ] D0 写 task63_sinkhorn_l1.py + 跑 L1 Sinkhorn vs vanilla
- [ ] D0 GO/NO-GO 判定 (PPL ≈ K 且 D_rel 劣化 < 10%)
- [ ] P1 完整 RQ 级联 (如 D0 GO)
- [ ] P2 η Pareto 扫描 (5 点)
- [ ] P3 Stage 3 TIGER 训练
- [ ] P3 Stage 4 推断 + Recall/NDCG 评估
- [ ] 写 verdict (含 result: 行 + vs Task #61 R@5=0.0977 对比)
- [ ] 更新 loop.md §16 (R8 归档)

---

## 7. 重叠审计

- ✅ 不与四战线重叠 (战线二修 embedding, G1 修量化器的分配几何)
- ✅ 不与已杀清单重叠 (QINCo 动态码本 ≠ G1 静态均衡约束; MMQ ≠ 边际质量守恒)
- ✅ 与幸存 idea "ambiguity-aware redistribution" 机制正交 (按解码歧义重分配 vs 按质量守恒约束分配), 可组合