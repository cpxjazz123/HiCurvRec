# 完全不依赖 Louvain、按节点度数匹配的结构分类实验

日期：2026-11-16  
对象：48 个标准 `3×(256,32)` VQ 码本 TIGER arm；q 从 checkpoint 码本 + 导出 SID 重建，k=10 全局 q-KNN。  
本实验**完全不使用 Louvain/社区标签**。

## 1. 分类与度数匹配

### 1.1 AF3 原始符号仍不可用

315,391 条真实行为边：

- AF3 > 0：**0**
- AF3 = 0：**1**
- AF3 < 0：**315,390**

AF3 随 dmax 强烈下降，因此分类使用 **AF3 在 dmax 五分位内的分位**，不使用绝对 AF3 阈值。

### 1.2 四类边

| 类 | 规则（不使用社区） | 边数 | 占比 | AF3 中位数 | dmax 中位数 | dmin 中位数 | asym 中位数 | common 中位数 | betweenness 中位数 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | T≥2 且度内 AF3 分位≥0.75 | 22,781 | 7.22% | -328 | 314 | 32 | 0.820 | 4 | 1.36e-8 |
| bridge | T=0，betweenness≥Q75，低 AF3 | 3,109 | 0.99% | -152 | 134 | 17 | 0.814 | 0 | 5.73e-8 |
| star | T=0，高度数不对称，高 dmax，低 dmin | 755 | 0.24% | -316 | 311 | 8 | 0.983 | 0 | 1.05e-7 |
| chain | T=0，低 dmax，低不对称 | 18,805 | 5.96% | -47 | 29 | 22 | 0.224 | 0 | 3.79e-9 |

类之间可重叠，不是互斥划分。

### 1.3 度数匹配

对每条边定义 `(dmax 五分位, dmin 五分位)` 共 25 个度数 cell。  
对每个类：

- `rate`：该类边的 q-kNN 恢复率；
- `matched`：同类度数 cell 中**非该类边**的恢复率，按该类边的 cell 分布加权；
- `excess = rate − matched`；
- `lift = rate / matched`。

这样直接控制端点度数的组成，而不是只把度数放进回归。

## 2. 结果

Partial 控制总体恢复率、q-KNN 边数、L3 collision。  
CV 是重复 5-fold（20 次，Ridge alpha=1）相对同一 base controls 的 R² 差。

| 类 | mean rate | mean excess | direct rho (rate) | direct r (rate) | partial rho (rate) | ΔR² rate | p(rate) | CV +rate | CV +excess |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 0.01676 | **+0.01131** | +0.660 | +0.748 | +0.043 | +0.0166 | 0.184 | +0.0011 | **+0.0175** |
| bridge | 0.00424 | **-0.00705** | +0.633 | +0.718 | +0.338 | +0.0564 | **0.012** | **+0.1403** | -0.0051 |
| star | 0.00149 | -0.00272 | +0.153 | +0.255 | +0.072 | +0.0131 | 0.239 | -0.0189 | -0.0173 |
| chain | 0.01019 | **-0.01624** | +0.686 | +0.755 | +0.246 | +0.0076 | 0.371 | +0.0210 | -0.0226 |

### 2.1 最重要的反差：bridge 的 raw rate 与 degree-matched excess

- **bridge raw rate** 的 CV 增量很大：**+0.1403**；
- 但 **bridge degree-matched excess** 的 CV 增量是 **-0.0051**；
- bridge 的 matched recovery 甚至高于 class recovery（mean excess -0.007）。

说明：

> **bridge 的“高 raw 恢复率”主要是端点度数/整体 q 质量带来的，而不是桥接结构本身。**

用户要求的度数匹配正好把这个假象去掉了。

### 2.2 dense

- **excess 均值 +0.0113**：dense 边确实比同度数 cell 的非 dense 边更容易被找回；
- in-sample excess ΔR² **0.0394（p=0.038）**；
- CV excess 增量 **+0.0175**，但 rate 增量只有 +0.0011；
- partial rate 只有 +0.043/+0.202。

所以 dense 有一个**很弱、方向不一致**的 degree-matched 信号；不能称为稳定预测因子。

### 2.3 star / chain

- star：excess 均值 -0.0027，CV excess -0.0173；
- chain：excess 均值 **-0.0162**，CV excess **-0.0226**；
- chain 的 raw rate 直接相关很高（+0.686/+0.755），但 degree-matched 后没有正增量。

chain 的高 direct correlation 也主要是度数/整体质量。

## 3. 结论

1. **完全不依赖 Louvain 的结构分类可以成立**：dense/bridge/star/chain 都能仅由 AF3 度内分位、共同邻居、度数不对称和 edge betweenness 定义。
2. **AF3 的绝对符号没有分类价值**：315,390/315,391 条边为负。
3. **度数匹配非常关键**：
   - bridge raw rate 的 CV 增量 +0.140 是**度数/整体质量假象**；matched excess 为 -0.005。
   - chain raw rate 的高相关也在度数匹配后消失（excess -0.016）。
4. **dense 有弱 degree-matched 信号**：excess 均值 +0.011，CV excess +0.0175，但 in-sample p=0.038、partial 很小，不能作为强结论。
5. 没有哪一类结构能提供**稳定、强、degree-matched 的独立 Recall 预测**；这与此前固定边数/CV 的结论一致。

## 4. 限制

- 48 个标准 VQ arm；26 个 mixed/hyperbolic/非标准 arm 未纳入。
- 度数匹配置信的是 `(dmax,dmin)` 五分位 cell，不是逐边精确匹配；同一 cell 内仍有残差度数差异。
- 类之间可重叠；bridge/star/chain 的规则使用了分位阈值，未做阈值敏感性扫描。
- edge betweenness 是 k=100 的近似值。
- CV base R² 只有 0.14 左右，fold 方差大；不能把单次 CV 差异当确定结论。
- 跨臂观察性，不是因果；没有 Stage3 重训。

## 5. 产物

- `analysis_sid_recall/qknn_degree_matched_structures.py`
- `analysis_sid_recall/results/qknn_degree_matched_structures.json`
- `analysis_sid_recall/results/qknn_degree_matched_structures.md`
