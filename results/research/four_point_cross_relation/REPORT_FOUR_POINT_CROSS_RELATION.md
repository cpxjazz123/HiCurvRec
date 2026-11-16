# 四点条件 × 交叉语义关系

日期：2026-11-17  
范围：离线 embedding/类别审计；不训练模型，不运行 Stage3。

## 1. 判据

对四个商品 A、B、C、D：

\[
S_1=d_{AB}+d_{CD},\quad
S_2=d_{AC}+d_{BD},\quad
S_3=d_{AD}+d_{BC}
\]

树度量要求 `S1,S2,S3` 中最大的两个相等（Buneman 四点条件）。本报告使用归一化 sentence-T5 向量的欧氏距离。

目标模式：

- A、B 同一 fine category；
- C、D 同一 fine category，且与 A/B 不同 coarse；
- A–C、B–D 接近；
- A–D、B–C 明显更远。

若同时出现，四点条件被违反，而且两条“近”关系分别来自**类别相似**和**功能互补/共同使用**。

## 2. 总体偏离

前一轮修正后的文本 embedding：

- 200,000 个无重复四元组中，**17.4%** 的 four-point residual > 0.05；
- 均值 residual 0.0291，中位数 0.0244；
- 超度量检验更差：50.6% 的三元组超过 0.05。

这说明文本距离整体上不是一棵精确树。

## 3. 显式交叉四点示例

以下是从**预先指定的互补类别对**中，在满足“cross pairs 严格近于 off pairs”的条件下选出的最佳候选。它们不是随机样本，不能用于估计总体比例。

| 关系 | A | B | C | D | S1 | S2 | S3 | residual |
|---|---|---|---|---|---:|---:|---:|---:|
| 电吉他 + 电吉他弦 | Epiphone Les Paul-100 | B.C. Rich Warlock Electric Guitar | Gibson Les Paul Premium Electric Guitar Strings | GHS GBUL Boomers Electric Guitar Strings | 1.222 | 1.212 | 1.462 | **0.1639** |
| 电容麦克风 + 麦克风线 | CAD Audio GXL3000 Condenser Mic | MXL-603S Instrument Mic | Hosa TRS-to-XLR Adaptor | Mogami Silver XLR Microphone Cable | 1.130 | 1.147 | 1.304 | **0.1203** |
| 尤克里里 + 琴盒 | Oscar Schmidt OU53S Baritone Ukulele | Kala KA-TE Tenor Electric Ukulele | Oscar Schmidt UC3 Concert Case | Kala Tenor Ukulele Gig Bag | 1.357 | 1.360 | 1.587 | **0.1426** |

这些四点的共同结构：

- A/B 同类：两把吉他；两支麦克风；两把尤克里里。
- C/D 同类：两套琴弦；一条转接线 + 一条麦克风线；两个琴盒。
- A–C / B–D 是跨类别的**功能互补**关系。
- S1、S2 接近且都小于 S3，树条件要求 S1=S2 或 S2=S3（最大两个相等），这里不成立。

例如电吉他例子：

\[
S_1=1.222,\quad S_2=1.212,\quad S_3=1.462
\]

最大两个是 `1.462` 和 `1.222`，差 0.240；相对 residual **0.164**。四件商品在语义上构成：

- 两把电吉他（同类相似）；
- 两套电吉他弦（同类相似）；
- 电吉他–弦（跨类互补）。

因此这不能由“乐器类 → 吉他类 → 电吉他”这一棵树的嵌套分组同时表达。

## 4. 候选生成中的其他例子

全类别对搜索还找到了大量“同一商品类型被 Amazon 分到不同分支”的四点，例如：

- 踩镲架 / 踩镲离合器 / 踩镲铜钹；
- 扬声器 / 扬声器零件；
- 鼓皮 / 鼓皮配件。

这些例子也说明类目树不是语义树的精确复制，但它们主要是**同一产品类型的类目拆分**，不是两种独立语义维度的交叉，因此不作为主要证据。

## 5. 结论

- 四点条件在文本 embedding 上确实有不可忽略的违反（17.4% > 0.05）。
- 能手工核验出至少三组**跨类别、功能互补**的四点结构；它们同时满足类别内接近和跨类别互补，且违反树度量。
- 这支持“商品语义不只是一棵层级树”，但仍是**小样本、类别对预先指定**的证据，不能估计总体普遍性。
- 不能从这些例子直接推出“混合曲率一定更好”；还需要证明这些关系有行为价值，并在量化前后比较。

## 6. 产物

- `results/research/tiger_cluster_graph/four_point_cross_relation.py`
- `results/research/four_point_cross_relation/four_point_cross_relation.json`
- `results/research/four_point_cross_relation/four_point_cross_relation.md`
- `results/research/four_point_cross_relation/four_point_cross_relation_candidates.csv`
