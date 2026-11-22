# VERDICT — Issue #237

**标题**:Stage1: 使用 GraphMoRE 自适应混合曲率 κ-GCN 生成商品增强 Embedding
**正文 sha256**:`8baddc18bf23a92bbf18736b84c435812ab5d4ce9c61253aa8e47da5db0265d4`
**是否需要训练**:是(issue 原文「本轮允许训练 GraphMoRE 表示学习模型」)

---

## 结论

**行为结构保留有明确改善(C28 的第一半成立),但语义保真几乎全部丢失;加上语义约束后结构
收益消失并跌破原始基线。两个目标在本数据集与配方下互斥。**

| variant | test AUC | test AP | 语义 cosine | kNN@10 |
|---|---:|---:|---:|---:|
| 原始 TIGER | 0.5957 | 0.6172 | 1.0 | 1.0 |
| **A 忠实 GraphMoRE(交付)** | **0.6562** | **0.6230** | 0.0046 | 0.0126 |
| B 加语义项(项目适配) | 0.5551 | 0.5714 | 0.8811 | 0.0699 |

---

## 逐条对账(C1–C29)

### A. 数据与图

| 编号 | issue 要求 | 实际 | 证据 | 判定 |
|---|---|---|---|---|
| C1 | 用 TIGER 原始 Embedding、冻结映射、#236 图,核验哈希 | 图 `6dbac454…`;原始 Embedding `6490c717…`(与 metadata 一致)| `export_manifest.json` | PASS |
| C2 | 图仅由 train 交互构建 | 构图源 `train.parquet`,传播图仅 train 边 | `metadata.policy`、`data.py` | PASS |
| C3 | CSR `data` 为 AF3,不当邻接权重 | 按 `indptr`/`indices` 构造,0 自环、对称 | `data.py::load_graph` | PASS |

### B. 模型

| 编号 | issue 要求 | 实际 | 判定 |
|---|---|---|---|
| C4 | 5 个两层 κ-GCN,初始曲率 {-3,-1,0,+1,+3} | 5 专家 × 2 层,0 冻结 | PASS |
| C5 | 多尺度 ego 子图 GCN + softmax 门控 | 2-hop/3-hop ego 指示池化 + 2 层 GCN + softmax | PASS |
| C6 | 专家在各自曲率空间聚合 | `logmap0 → GCN 归一化和 → expmap0`,流形聚合 | PASS |
| C7 | 节点权重 + 流形加权/拼接 + 权重对齐距离 | `Σ softmax(w_u⊙w_v)[e]·‖z_u^e−z_v^e‖²` | PASS |
| C8 | 核对官方代码,不把欧氏冒称原方法 | REPORT §3 逐项对照表 + 差异声明 | PASS |
| C9 | 无向图,不声称方向,不加有向 GNN,不按 Graphlet 指定曲率 | 0 处有向实现;三处声明 | PASS |
| C10 | #236 曲率仅用于分析 | 只进评估分层,不参与训练 | PASS |

### C. 训练目标与切分

| 编号 | issue 要求 | 实际 | 判定 |
|---|---|---|---|
| C11 | 链接预测 + 结构距离失真;明确归一化/温度/采样/距离/权重 | REPORT §4 全项 | PASS |
| C12 | 负例不宣称真实无交互 | 明确为"未观测对" | PASS |
| C13 | 拆 train/val/test 边;传播/ego/最短路/负例只用训练可见信息 | 264,929/25,231/25,231;全部只用 train | PASS |
| C14 | 固定划分与 seed 42,不做多种子 | seed 42,单次 | PASS |

### D. 输出

| 编号 | issue 要求 | 实际 | 判定 |
|---|---|---|---|
| C15 | 每商品有限、可复现的 768 维 | 24,587×768,全有限,重算 delta 0.0 | PASS |
| C16 | 记录切空间映射/拼接/投影并验证导出向量 | 5×128 → LN → Linear(640→768);导出得分保留原生 98.8% | PASS |
| C17 | 额外语义目标须标明为项目适配 | variant B 明确标注,默认关闭 | PASS |
| C18 | 原始 Embedding 独立文件不覆盖 | 原始 sha 未变,写入独立文件 | PASS |
| C19 | 孤立商品 fallback;覆盖全部;顺序对应 ID | 覆盖 24,587;顺序即 stage0 商品 ID;孤立点有限 | PASS |
| C20 | 冻结最佳验证 checkpoint,记录导出与哈希 | `best_checkpoint.pt`(epoch 395)+ 哈希 | PASS |

### E. 评估与交付

| 编号 | issue 要求 | 实际 | 判定 |
|---|---|---|---|
| C21 | 同冻结划分六项指标,同一评估器 | 语义 3 项 + AUC/AP + 失真 ρ + graphlet,同一 `evaluate.py` | PASS |
| C22 | 区分原生混合空间与导出 768 维得分 | 原生 0.6640 vs 导出 0.6562,分列 | PASS |
| C23 | 隔离 graphlet 评估;说明哪些只证无向 | bank 隔离;三处声明 fork/merge 不可区分 | PASS |
| C24 | bank 无法满足隔离则先报告 | bank 与传播图无交集(传播用 train 边,评估用 bank) | PASS |
| C25 | 专家初/终值、权重、使用率/熵、度数分层、曲率分层;查坍缩;不宣称因果 | REPORT §6 全表;坍缩已报告(±1 专家 0% 使用) | PASS |
| C26 | 代码/配置/日志/checkpoint/768 维/ID 映射/哈希/评估表/REPORT | 全部交付 | PASS |
| C27 | 导出与 checkpoint 重算一致;有限/维度/覆盖/无重复错位 | delta 0.0;24,587 维 768 全有限无重复 | PASS |
| C28 | **成功判据**:结构改善 + 报告语义代价;若只有原生提高则判未实现 | 结构改善 **+0.0605 AUC**;语义代价已报告;导出同样提高 | PASS |
| C29 | 保持 768/32/256×3;不启 Stage2/Stage3;不重训 TIGER;不覆盖;不建 Issue | 0 处 | PASS |

**29/29 PASS。**

---

## 关键数字

### 行为结构(test,冻结划分)

| 指标 | 原始 | 增强 | 变化 |
|---|---:|---:|---:|
| AUC | 0.5957 | **0.6562** | **+0.0605** |
| AP | 0.6172 | 0.6230 | +0.0058 |

### Graphlet 排序率(无向投影)

| 结构 | 原始 | 增强 |
|---|---:|---:|
| chain | 0.3698 | **0.3802** |
| fork | 0.4228 | **0.4362** |
| merge | 0.4259 | **0.4753** |

### 语义代价

| 指标 | 增强 |
|---|---:|
| cosine 均值 | **0.0046** |
| ridge R² | 0.3778 |
| kNN@10 重叠 | **0.0126** |

### 曲率

初始 {-3, -1, 0, +1, +3} → 终值 **{-3.266, -1.144, 0, +1.012, +3.409}**

### 门控

| 专家 | 平均权重 | argmax 使用率 |
|---|---:|---:|
| κ=−3 | 0.521 | 60.0% |
| κ=−1 | 0.001 | **0.0%** |
| κ=0 | 0.234 | 16.5% |
| κ=+1 | 0.0005 | **0.0%** |
| κ=+3 | 0.244 | 23.5% |

熵 0.689 / 1.609。**±1 两个专家从未被选中**——5 专家配置过参数化。

---

## 必须声明的限制

1. **`geoopt` 未安装**,κ-stereographic 运算自行实现;`mobius_matvec` 用切空间线性形式
   (代数等价、非闭式),优化器用 Adam 而非 `RiemannianAdam`。已逐项对照官方代码。
2. **ego 子图上限 64/128 节点**(图比论文基准大 9 倍);池化数学等价,但子图规模不同。
3. **无向投影**:不声称保留方向;fork 与 merge 在无向投影上同构,无法区分。
4. **门控与度数/曲率的相关是相关性**,不是因果;训练信号是链接预测,不是"按曲率分配专家"。
5. 单次运行 seed 42,无多种子;因此 AUC 差值的置信区间未估计。

---

## 证据索引

| 内容 | 路径 |
|---|---|
| 报告 | `results/research/issue237/REPORT.md` |
| 交付向量 | `enhanced_768.npy`(sha `0f3c0d1d…`,24,587×768) |
| 专家切空间向量 | `expert_tangent.npy`(24,587×640) |
| 门控权重 | `gating_weights.npy`(24,587×5) |
| 商品 ID | `item_ids.npy` |
| checkpoint | `best_checkpoint.pt`(sha `3346af1b…`,epoch 395) |
| 导出清单 | `export_manifest.json`(全部哈希 + 一致性检查) |
| 评估 | `evaluation.json` |
| 训练日志 | `train_log.jsonl`(400 epoch) |
| variant B | `results_semantic/`(全部产物) |
| 代码 | `graphmore/{manifold,model,data,train,export,evaluate}.py` |
| 测试 | `tests/test_manifold.py`(43)、`tests/test_model.py`(9),**52 项全绿** |
| 清单与 infra | `RUN_LOG.md`、`INFRA.md` |
| issue 评论 | `COMMENT_1.md`、`COMMENT_2.md`、`COMMENT_3.md` |
