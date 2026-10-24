# ROLE M — mechanism declaration

## Status: WITHDRAWN

我无法依据当前工作区可见的冻结证据负责任地提名机制。这是撤回候选，不代表所有可能机制都已被证伪；我不会编造触发频率，也不会声称已经测得频率。

## 冻结协议

不改变 32 维潜变量、三个各含 256 个码的码本、三个整数 SID token、原数据集/切分以及 Stage2/Stage3 配方。三个严格门槛仍为 Hard-L1 ER > 16.38953942232631、Stage3 Recall@10 > 0.06406796775709883、NDCG@10 > 0.03530651701940677。

## OPERATOR

无。没有提议公式、定义域、曲率、基点或切空间语义。未从冻结 artifacts 确立其作用配置的实际频率前，命名算子会违反 6c。

## FINITE-PRECISION DEFINITION

不适用：未提议算子。因此不存在留待实现决定的 epsilon、clamp、比较容差或边界处理。

## GRADIENT PATH

无。不提议梯度、stop-gradient 边界或损失项变更。

## INVARIANTS THAT FAIL FOR THE IDENTITY

不主张任何此类不变量。没有明确的非恒等算子，声称存在恒等算子所不具备的不变量将缺乏依据。

## PRE-REGISTERED FALSIFIERS WITH NUMBERS

不登记候选专属 falsifier。三个原有严格门槛保持不变；只通过其中部分门槛不够。要求的冻结 artifacts 触发率数值目前**不可得**，不是零：当前工作区只有 `opencode.json`，不含计算分子和分母所需的 assignment stream / 冻结 Stage2 artifacts。因此无法证明作用配置是常见事件。

## CAPACITY STATEMENT

不增加可学习参数。不提议 layout 变更或 frame transport。这是未提案声明，不是候选容量主张。

## WHY NOT THE SIX CLOSED FAMILIES

不复活或改名包装任何已关闭家族。本声明不提议按码字半径做 product-geodesic level-0 assignment、horospheric/Busemann level sets、codeword-frame transported residual、role-assigned token layout、gyro/Möbius associator scoring、unsigned tangent wedge area 或 LOTV tie-break。尤其不提议任何未经频率审计的稀有配置 tie-break。

## WHY NOT HiHPQ (arXiv 2401.07212, abstract-only)

不对 HiHPQ 提出新颖性或不重叠主张。现有材料仅为摘要，且本声明没有候选算子可供比较。仅有摘要不能代替算子层面的比较。

## AUDIT

不得依据本声明启动实验。后续提案前，须先提供冻结 assignment stream（含 item/sequence 标识及三个 level assignments）、冻结 Stage2 embeddings/codebooks 及数值格式，以及精确 artifacts manifest/checksums。预提案审计须输出事件谓词、分子、分母、频率、聚合规则和确定性的 tie/boundary 处理，使频率可在不训练的情况下重算。只有报告该频率后，方可声明候选；候选运行还须输出重现决策、梯度及三个评估指标所需的所有算子操作数与中间量。

**Decision：WITHDRAWN**，等待足以计算常见配置频率的冻结 artifacts 证据。不作性能主张。
