# ROLE M — WITHDRAWN

## 决定

**WITHDRAWN：本轮不提出新机制。** 当前可见冻结材料中没有 assignment stream、可重放该 stream 的冻结输入/候选记录，或既有频率审计报告。因此无法按要求给出机制作用配置在冻结工件上的实测发生次数与比例。发生率不能以“预期常见”、理论概率或未来训练后再测代替；在测量之前提出候选会违反冻结协议 §6c。

本文件不构成实验候选，不授权实现、训练或修改协议。没有把任何示例算子作为候选。

## 缺失的频率证据

```text
eligible_count = 尚不可计算
eligible_rate  = 尚不可计算
```

冻结工件目录中未提供给本工作区 assignment stream 或其可重放输入。现有候选说明（包括已撤回的 LOTV）不能提供本次不同机制所需的频率测量。故不报告虚构的 count、分母、百分比或抽样估计。

解除阻断前，必须只读冻结工件、无需训练地计算并发布完整 stream 分母、逐样本适用条件、符合条件的计数及比例、解析脚本/版本与输入哈希、独立重算结果。只有实测比例证明触发情形并非罕见后，才可另行提出机制。不得事后放宽阈值把罕见情形称为常见。

## OPERATOR

**不适用（WITHDRAWN）。** 本轮没有被提名的算子、定义域、曲率、基点或切空间语义。不能在缺少频率证据时借写出数值良好的公式绕过 §6c。

## FINITE-PRECISION DEFINITION

**不适用（WITHDRAWN）。** 没有候选算子，因此不声明 epsilon、clamp、比较容差或退化边界规则。未来提案必须在声明中逐项固定这些规则及所有边界行为；仅在精确算术下成立不合格。

## GRADIENT PATH

**不适用（WITHDRAWN）。** 没有前向算子或训练目标变更。

## INVARIANTS THAT FAIL FOR THE IDENTITY

**无可声明的不变量。** 未提出算子，故不声称存在区别于冻结 identity 的可检验行为。

## PRE-REGISTERED FALSIFIERS WITH NUMBERS

无候选，故不注册会被误解为可执行实验的候选阈值。任何未来候选至少必须在训练前通过：

1. **发生率门槛：** 从冻结 assignment artifacts 完整枚举并报告适用计数/比例；比例须先证明不是罕见情形，否则拒绝候选。计数、分母、脚本和输入哈希必须可独立复算。
2. **有限精度门槛：** 预先固定有限精度、epsilon、clamp、比较容差、非有限值处理和所有退化边界行为；数值 oracle 不通过即拒绝。
3. **协议门槛：** 32 维潜变量、恰好 3 个 256-entry codebooks、每个项目恰好 3 个整数 SID token、原数据/切分以及 Stage2/Stage3 recipe 全部保持不变。
4. **最终成功门槛：** 同一冻结数据、切分和未修改 recipe 下，必须同时严格超过 Hard-L1 ER `16.38953942232631`、Stage3 Recall@10 `0.06406796775709883`、NDCG@10 `0.03530651701940677`；任一未严格超过即失败。

## CAPACITY STATEMENT

没有机制、参数、布局或 frame transport 被提出或实现。本文件不变更任何容量或协议设定。

## WHY NOT THE SIX CLOSED FAMILIES

本轮没有新候选，故不以改名、变体或拼接方式重提任何已关闭家族：

1. 不重提 product-geodesic level-0/codeword-radius assignment。
2. 不重提 horospheric/Busemann level sets。
3. 不重提 codeword-frame transported residual 或其恒等式形式。
4. 不重提 role-assigned token layout。
5. 不重提 gyro-composition/Möbius associator score。
6. 不重提 unsigned tangent-space wedge；也不复活已撤回且未先证明 tie rate 的 LOTV tie-break。

## WHY NOT HiHPQ (arXiv:2401.07212, abstract-only)

本轮没有候选可与 HiHPQ 比较。现有信息仅为 abstract-only，不能据此声称新颖性、非重叠或该论文未覆盖某机制。未来提案必须先取得并审阅全文及补充材料，再对具体算子、适用配置和训练/赋值位置逐项比对；在此之前不作新颖性主张。

## AUDIT

本轮不运行候选，不请求修改训练代码。频率审计要让第三方可从冻结材料重算，至少须输出：

* 冻结数据集/切分 manifest、Stage2 checkpoint/config、recipe、源代码版本及其哈希；
* assignment stream 完整样本数、样本稳定 ID、逐样本现有 SID/候选及产生它们所需的原始操作数；
* 预先定义且不依赖训练结果的适用条件，以及每个样本的适用/不适用判定和原因；
* `eligible_count`、确切分母、`eligible_rate = eligible_count / denominator`，不四舍五入后再作稀有性判断；
* 用于枚举和计数的脚本、运行环境/数值精度、日志及输入哈希；
* 独立 checker 重建计数和比例的结果。

在以上工件缺失时，频率结论保持“不可计算”，不得据此启动机制实验。
