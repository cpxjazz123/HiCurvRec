# Task #337 / Issue #50 Gate 0' — 真实噪声底线测量（取代 σ 拍脑袋档位）

> 来源: [Issue #50] (https://github.com/WENYULIANG123/GeneRec/issues/50)
> 关联: Issue #48 / Task #339 (被质疑的 Gate 0 方法论)
> 关联: Issue #49 (δ=0.02 取决于本任务测得的真实噪声量级)

## 1. 背景（Issue #50 主张）

Issue #48 Gate 0 (Task #339) 推翻 H2 时用 σ={0.01, 0.02, 0.05} 三档**人为选定的** Gaussian 噪声，未验证这些 σ 是否代表真实商品向量波动幅度。Issue #50 主张必须先测真实噪声，再判定 H2。

## 2. Gate 0' 方法设计（与 Issue #50 § 实验设计对齐）

### 方案 B（最优先，确定性实现）
- 找 N=200 个 Musical_Instruments "近似重复"商品对：
  - 同 brand + 描述 ≥ 0.85 Jaccard 相似度
  - 或同 ASIN / 同一 listing 的不同变体
- 计算每对在 3 层 residual (r0, r1, r2) 上的欧氏距离分布
- 得到 **真实残差基线**：per-layer (mean, std, p5, p50, p95)

### 方案 A（次优先）
- 当前 embeddings (`item_emb.parquet`) 已经是 deterministic T5-base 编码
- 方案 A 假设若 encoder 引入 dropout → 重复 N 次测量方差
- T5-base inference mode 默认确定性 → 测出来方差≈0，需 fallback 到方案 B

### 方案 C（最后）
- 同义替换 + drop-adjective 文本扰动后重新编码
- 测传播到 residual 的方差

## 3. 通过条件

得到 per-layer 真实噪声量级（mean + percentiles），对照 Task #339 Gate 1 码字间隔:
- L0 NN dist p5 = 0.1004
- L1 = 0.0630
- L2 = 0.0425

判定 H2 真伪：
- 真噪声 ≪ 码字间隔 → H2 仍成立（Issue #49 δ 保持 0.02）
- 真噪声 ≈ 码字间隔 → H2 边界成立（δ 需要重新校准）
- 真噪声 > 码字间隔 → H2 推翻结论加强（Issue #49 δ 可能需要更激进）

## 4. 实现步骤

1. **方案 B 数据准备**：从 `HG-Rec/dataset/Instruments/` 找近似重复商品对
   - 用 `title` + `brand` + `description` 的 Jaccard 相似度筛选
   - 或同一 ASIN 多 color/size 变体
2. **方案 A 测量（if applicable）**：重复编码测方差
3. **方案 B 测量**：配对残差分布
4. **方案 C 测量**：文本扰动 + 重新编码
5. **写 verdict** `verdicts/task337_issue50_result.md`
6. **commit + push + close Issue #50**

## 5. GPU 占用

- 主 CPU 端: 数据相似度计算 + 残差聚合
- 少 GPU (单卡 ~5GB): 方案 A/C 的 T5-base 重复编码（~30 min）

## 6. 决策阈值

| 项目 | 阈值 |
|------|------|
| 数据驱动噪声量级 | mean + std + p5/p50/p95 per-layer |
| 三方案至少一个产出有意义分布 | 否则如实记录"无法可靠测量" |
| H2 重新判定 | 写进 verdict，与 Task #339 对照 |

## 7. 完成判定

- `verdicts/task337_issue50_result.md` 存在 + 含 `result:` 行
- 至少方案 B (确定性最高的方案) 完成
- commit + push + Issue #50 closed
