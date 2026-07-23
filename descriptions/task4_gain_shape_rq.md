# Task 16 (Gain-Shape RQ)：残差的"大小"vs"方向"哪个携带更多下游信息

> **状态**: 🟡 backlog

> **目的**：检验 Gain-Shape Residual Quantization 的核心假设——**残差的方向（unit vector）比大小（norm）携带更多对下游任务有用的信息**。
> **数据集**：Amazon Toys（11924 商品，含 category 标签元数据 + 协同过滤信号）
> **目标仓库**：`/fs04/ar57/wenyu/GeneRec/GRID`
> **环境**：`conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys`

---

## 总目标

Gain-Shape RQ 论文主张：把残差向量拆成 **magnitude (gain, 标量)** 和 **direction (shape, 单位向量)** 两部分，**对两者分别量化**。理由是：方向比大小更稳定、更携带语义信息。

**核心假设**：方向向量对下游任务（品类预测 / 协同信号预测）的预测能力，应该**显著高于**大小标量的预测能力。

---

## 具体做法

### 输入
- 第 3 层（最深层）残差向量 `(N=11924, D=2048)`：从已训练好的 RKMeans / RVQ / RQ-VAE 取出
- 商品元数据：
  - **品类标签**（如果 Toys metadata 有）：每个商品的 category 或 brand（多分类）
  - **协同过滤信号**：训练数据中**高频共现商品簇 ID**（从 user-item 交互矩阵聚类得到，例如 KMeans on user vectors → 商品簇标签）
- 备用：若 Toys 元数据无 category 字段，可从 Amazon meta JSON 文件解析

### 计算步骤
1. 取出第三层残差矩阵 `R ∈ R^(N, D)`
2. 拆分大小和方向：
   - `gain = ||R[i]||` (N,)：每个商品的残差范数
   - `shape = R[i] / ||R[i]||` (N, D)：每个商品的残差单位向量
3. 准备 4 种"残缺版本"的输入：
   - `R_full` (N, D)：完整残差（基线）
   - `R_mag_only` (N, D)：把 shape 替换为常数向量（如全 1）只保留 gain 信息（即 `R_mag_only[i] = gain[i] * 1_vec`）
   - `R_shape_only` (N, D)：把 gain 归一为常数（如 1）只保留 shape 信息（即 `R_shape_only[i] = shape[i] * 1`）
   - `R_zero` (N, D)：全零（纯噪声基线）
4. 训练 4 个轻量线性分类器（Logistic Regression / MLP 1 层）：
   - 输入：上述 4 种残差特征
   - 目标：商品 category 标签（如果有）或共现商品簇 ID
   - 评估：accuracy / NDCG / F1
5. 重复 5 次（不同随机种子），取平均

### 关键判断

| 结果模式 | 含义 | Gain-Shape RQ 价值 |
|---------|------|---------------------|
| `shape_only` 准确率 ≫ `mag_only` 准确率 | 方向携带大部分信息 | **高**：支持 Gain-Shape 拆分 |
| `mag_only` ≫ `shape_only` | 大小更重要 | **低**：方向分离无意义 |
| 两者差不多 | 大小和方向同等重要 | **中**：拆分有边际收益但不显著 |
| `full > mag_only ≈ shape_only` | 信息冗余但叠加有价值 | **中**：完整残差仍有优势 |

**判断标准**（至少满足其一才算"支持 Gain-Shape"）：
1. `acc(shape_only) / acc(mag_only) > 1.5`（方向预测能力明显更高）
2. `acc(full) - acc(shape_only) < 0.05 * acc(full)`（方向已经捕捉大部分信息，加大小增益小）

---

## 实现要点（GRID 代码定位）

提取第三层残差（同 exp14）：
```python
# 加载 RKMeans model，做逐层量化
r = x.clone()
for l in range(L):
    r = quantize_one_layer(r, codebooks[l])
# 循环结束后 r 即第三层残差
```

拆分大小和方向：
```python
gain = r.norm(dim=1, keepdim=True)            # (N, 1)
shape = r / (gain + 1e-8)                     # (N, D)
```

训练线性分类器：
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# labels: 商品 category ID (N,)
X_full = r.cpu().numpy()
X_mag = (gain * torch.ones_like(shape)).cpu().numpy()
X_shape = (1.0 * shape).cpu().numpy()
X_zero = np.zeros_like(X_full)

for name, X in [("full", X_full), ("mag_only", X_mag), ("shape_only", X_shape), ("zero", X_zero)]:
    X_tr, X_te, y_tr, y_te = train_test_split(X, labels, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=200, multi_class="multinomial")
    clf.fit(X_tr, y_tr)
    acc = clf.score(X_te, y_te)
    print(f"{name}: acc={acc:.4f}")
```

需要 Toy 品类数据：
```bash
ls /fs04/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys/
# 看看有没有 meta_*.json 或类似的元数据文件
```

---

## 完成指标

| 指标 | 目标 | 验证 |
|------|------|------|
| 第三层残差 shape | (11924, 2048) | `torch.load` |
| 4 种特征 + 训练脚本 | `train_classifier.py` | bash + py_compile 通过 |
| 4 种 acc 结果 | 1 个 JSON 报告 | 验证 |
| 显著性检验（5 次平均 ± std） | 输出 | 验证 |

---

## 风险与回退

1. **Toys 元数据无 category 字段** → 改用协同过滤信号：从 `data/amazon_data/toys/training/*.tfrecord` 提取 user-item 交互矩阵 → KMeans 聚类得到商品簇 ID 作为替代标签
2. **残差 D=2048 维太高，Logistic Regression 慢** → 先 PCA 降维到 64 维再训练
3. **结果不显著** → 改用 MLP 1 层（线性分类器可能太弱），或换用 contrastive learning proxy（同一用户商品对的残差应该相似）

---

## 与 Gain-Shape RQ 论文的关联

Gain-Shape RQ 主张：
- 把 `r ∈ R^D` 拆成 `r = g · s`（gain × shape）
- 对 g（标量）和 s（D 维单位向量）**分别量化**
- 优点：方向（shape）量化误差小，因为它在 unit sphere 上更"紧凑"

**本实验是 Gain-Shape RQ 的"信息分离假设"检验**：
- 如果 shape 比 mag 信息量大 → 拆分有意义
- 如果 mag 和 shape 信息量相当 → 拆分只是实现细节，没有理论收益
- 如果 mag 信息量 > shape → Gain-Shape RQ 假设可能错了，应该用其他变体

---

## 备注

⚠️ Gain-Shape RQ 还有另一层含义——**量化算子设计**（用球面量化 vs 标量量化），本实验只测"信息分布"层面，不涉及量化算子本身的实现。

如果要完整复现 Gain-Shape RQ 的量化算子，需要修改 GRID 的 RKMeans / RVQ 类（增加 `gain_quantizer` 和 `shape_quantizer` 两个独立 codebook），这是更大的工程。本实验只是先做可行性验证。