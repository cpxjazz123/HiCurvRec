# Task #66 — G3: 量化的拓扑保真度 (TDA) — QMP 看不见的破坏维度 (高风险高辨识度)

> **任务目的**: 验证 G3-H1: 两个量化器可以有相同的平均度量失真 D_rel, 但对 embedding 空间拓扑结构 (连通分支 H_0, 环路 H_1) 的破坏程度完全不同; 拓扑破坏捕获 QMP 全指标遗漏的失败模式
> **执行日期**: (待启动, 优先级 5/5, 风险最高)
> **状态**: 🟡 待启动

---

## 1. 背景

承接 Task #58-#61 Campaign + G1/G2/G5. QMP 是度量统计, G3 是拓扑统计, 数学上正交. 这可能是 Task #85/#87 "邻域保持必要不充分"悖论的缺失变量之一 — 两个 Hamming lift 相近的 tokenizer, 拓扑破坏可以差一个量级.

**核心数学**:
- Vietoris–Rips 过滤 $\mathrm{VR}_\epsilon(X) = \{\sigma \subseteq X : \mathrm{diam}(\sigma) \le \epsilon\}$
- Persistence diagram $\mathrm{Dgm}_p(X)$ (p 阶同调特征的出生-死亡对)
- Bottleneck 距离: $d_B(\mathrm{Dgm}_p(E), \mathrm{Dgm}_p(\hat{E})) = \inf_\eta \sup_u \|u - \eta(u)\|_\infty$
- 稳定性定理: $d_B \le \|d_E - d_{\hat{E}}\|_\infty$ → $d_B$ 是最坏情形结构失真的下界见证, 与 $D_{\mathrm{rel}}$ 平均失真**数学上不同**

**风险声明**: 这是组合里最可能"测出一堆噪声"的方向. D0 不过立即停, 沉没成本 3 天. 但若成立, 辨识度是五个方向里最高的 (社区没人有这个视角).

---

## 2. 实验设计

**变量**: 量化方式 (标准 VQ → 拓扑正则化 VQ)
**保持不变**:
- Stage 1 embedding (复用 Task #59/#60/#61 多源)
- seed=42, K=256

**启动命令**:
```bash
# D0: 2000 点子样 H_0/H_1 diagram (2-3 天, 需装 Ripser)
pip install ripser  # 一次性

python3 scripts/task66_persistence.py \
    --embeddings logs/task59_s1/.../merged_predictions_tensor.pt \
                 logs/task61_s1/merged_predictions_2816d.pt \
    --rq_outputs logs/task59_s2_infer/pickle \
                 logs/task61_s2_infer/pickle \
    --n_subsample 2000 --n_bootstrap 5 --max_dim 1 \
    --out_json verdicts/task66_persistence.json

# P1: 全数据 + Ripser H_1 (5 天)
python3 scripts/task66_full_tda.py \
    --embeddings [Task #58/#59/#60/#61/#87 路径] \
    --rq_outputs [对应 RQ 路径] \
    --max_dim 1 --n_landmarks 500 \
    --out_csv verdicts/task66_full_tda.csv

# P2: 可微拓扑正则化 (5 天)
python3 scripts/task66_topo_regularized_vq.py \
    --input_pt logs/task59_s1/.../merged_predictions_tensor.pt \
    --output_dir logs/task66_s2/pickle \
    --K 256 --num_hierarchies 3 --topo_beta 0.1 --seed 42

# P3: 端到端 TIGER
CUDA_VISIBLE_DEVICES=1 python3 -m src.train experiment=tiger_train_flat \
    data_dir=data/amazon_data/toys \
    semantic_id_path=logs/task66_s2/pickle/cluster_ids.pt \
    sequence_length=120 num_hierarchies=4 task_name=task66_s3_train
```

---

## 3. 决策触发 (vs 提案 D0 GO 条件)

| D0 diagram 差异 | 决策 |
|-----------------|------|
| diagram 差异 > 自举噪声带 (5 次 bootstrap 标准差) | ✅ GO → P1 全数据 + P2 拓扑正则 + P3 TIGER |
| diagram 差异接近噪声带 | ⚠️ PARTIAL → 改用 witness 复形 + 增加 landmark 数, 或换 GPU TDA 库 (e.g., Flagser) |
| diagram 差异 = 噪声 (T5 vs MCKG 重构后拓扑无显著差异) | ❌ NO-GO → 关闭 G3 方向, 止损 3 天 |
| P3 R@5 ≥ 0.0977 (Task #61) | ✅✅ 拓扑正则化端到端胜利 |
| P3 R@5 < 0.0977 | ⚠️ 拓扑诊断成立但解法未带来下游收益, 留作 verdict 反例 |

---

## 4. 预算

| 阶段 | 估算时间 | GPU |
|------|---------|-----|
| P0 文献核查 (1 周) | 1 天 | 无 |
| **D0** 子样 H_0/H_1 diagram | **2-3 天** | 无 (Ripser CPU, 子样 2000 点) |
| P1 全数据 TDA + Ripser H_1 | 5 天 | 无 (但需子采样/witness, 计算密集) |
| P2 可微拓扑正则化 | 5 天 | cuda:0 (~2 hour) |
| P3 Stage 3 TIGER | 3 天 | cuda:1 (~120 min) |
| P3 Stage 4 + 评估 | 5 min + 1 min | cuda:1 |
| **总计** | **~17 天** | **~5 GPU-hour** |

---

## 5. 风险与缓解

**风险 1 (最高)**: D0 测出一堆噪声 → **止损机制**: D0 阶段 3 天内若 diagram 差异 ≤ bootstrap noise, 立即停止, 写 verdict NO-GO, 资源转 G1/G2/G5
**风险 2**: $H_1$ VR 复杂度对单纯形数爆炸 → witness / landmark 复形 (maxmin 采样 ~500 landmark) + Ripser sparse filtration
**风险 3**: MST 边选择不可微 → 用 straight-through estimator 或 Gumbel-softmax 软选边
**风险 4**: 拓扑正则化与覆盖率/失真三方权衡 → P2 阶段扫 topo_beta ∈ {0.01, 0.05, 0.1, 0.5}, 报 Pareto

---

## 6. 完成度跟踪

- [ ] P0 文献核查 (检索词 `topological data analysis vector quantization`, `persistent homology recommendation`, `topological autoencoder discrete codes`)
- [ ] D0 装 Ripser + 写 task66_persistence.py + 跑子样 H_0/H_1
- [ ] D0 GO/NO-GO 判定 (diagram 差异 > 自举噪声带)
- [ ] **止损检查**: D0 不达 GO → 立即停, 不进 P1
- [ ] P1 全数据 TDA (如 D0 GO)
- [ ] P2 可微拓扑正则化
- [ ] P3 Stage 3 TIGER 训练
- [ ] P3 Stage 4 推断 + Recall/NDCG 评估
- [ ] 写 verdict + 更新 loop.md §16

---

## 7. 重叠审计

- ✅ 与四战线零重叠 (战线一经验度量 vs G3 拓扑)
- ✅ 与已杀清单零重叠 (无已杀 idea 涉及拓扑)
- ✅ 与 QMP 数学上正交 (度量统计 vs 拓扑统计)
- ⚠️ 与 G2 几何场有部分方法重叠 (kNN 图构建), 但目标函数不同 (G2 局部曲率 vs G3 全局拓扑特征), 可共享 kNN 计算代码