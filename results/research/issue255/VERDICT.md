# Issue255 verdict — BLOCKED_BY_FROZEN_COLLISION_GATE

## 判定

**工程实现与所有可执行检查、四卡3000epoch及全量机制评价已交付；本issue不得宣称COMPLETED。** 修正版 `issue255_formal_r1` 的三码碰撞率为0.9814129417985114，实际Stage3 MCP按冻结 `<0.4` 门槛拒绝启动。继续完成原合同的150epoch Stage3、最终test、配对与完整R2四件套，必须放宽冻结协议；没有该授权，也不允许追加正式训练追分。按 complete-open-issue §6.1“继续必须违反硬规则”关闭为 **BLOCKED**，不是工作量、等待时间或可修复bug导致的退出。

科学结论：本次组合机制产生技术可解码但不满足准入条件的塌缩SID。不能把92.57%的原始边共享率当成结构收益：严格test度匹配对照93.54%，高于direct关系91.87%。三级宏曲率MAE变差、微MAE改善，完整保留这个分歧。**推荐收益未检验，不能写成Recall下降或提升。**

## 冻结身份

- Issue正文SHA-256：`2fa23b1ee2faa897155f2a12999b3cf4cd7c085a5fa0a5f8a99c53f54b1c361d`；C1–C13原文见 `ACCEPTANCE.json`、`CHECK_4.md`。
- 数学合同：`CONTRACT.md:a47efec99b0dff2e03adcfdbb113af925693927736382630e5a1d496de7eb7d5`。论文/官方实现继承对照：`results/research/issue253/CONTRACT.md:2980f4f30017853f47dde1194c3e321a9cf5f90dc0c13b7e37c9cf1e7673154e` §59–70；新门控与量化不是GraphMoRE官方模块。
- 已推送生产实现commit：`3a38c3c9ea9623b6b9edae24585bf11a1225ae2a`。实际训练源为 `results/stage2_RQ-VAE/_runs/issue255_formal_r1/TIGER_prefix_curvature_gate_rq/src`，不是随后可能移动的活跃树。
- 最终checkpoint：`rqvae_best.pth:0969f60eaf5bde2b715bad99f86f5a4ab30ef63e2f7c9998ca8a710761ace2a9`；SID：`item_sids.json:7140826dbce9961f032818820e6fa38926e9fdd86473403151932ae22714b59b`；归档绑定：`archive_binding.json:179dc5c0de3d3860fcde8a16db529ed97947c889b7d03846373ae327b638134b`。
- 原 `issue255_formal` 为 `INVALID_IMPLEMENTATION`，保留但不用于结论。owner明确批准的纠错代次与原因见 `OWNER_APPROVAL.json`、COMMENT_07/08，不是低分重跑。

## 生产代码绑定

下列路径均相对上述实际 `src/`；分析工具相对仓库根。每条PASS的完整源/输入/命令/退出码绑定保存在对应JSON及 `EVIDENCE_MANIFEST.json`，不以参考实现或loss下降替代生产证据。

| 代号 | 路径 | SHA-256 |
|---|---|---|
| S2 | `model/structure.py` | `1bccb9f7dbf54f62a743258efc7faca5fc7f89c9cc0039b793c7f0e1985a126b` |
| S3 | `model/structure_gate.py` | `ce043cd2c395b45c9074e195547bd06ceb6915f2078c8f19781b049a4c209499` |
| S4 | `model/layer_gate.py` | `18bfc476986946bf47a5b80571630a3ef18b7653ff90856575d1240563421579` |
| S5 | `model/mechanism.py` | `7cea8be51adcaee041da2035d2b8285d46488f8197f4460726a13d0c92da1c28` |
| E | `tools/issue255/evaluate_arm.py` | `10a54765114451cce47e7a1dd2ad244d90a5838ff9fc0c9faff3727ea384d303` |
| X | `tools/issue255/extract_catalogues.py` | `fd1169db9866d71d59f5db5e7e7d8fb3615323ec3f1d28b846903f81c40f39a2` |
| R | `tools/issue255/reconcile_results.py` | `b97426edd7c7acd7f829830830176d4ec7c9eed8981e5a088fbbba5f83311204` |

## C1–C13 对账

| 条目 | 冻结要求摘要 | 实测/证据 | 判定 |
|---|---|---|---|
| C1 | 原容量、实际分配协议与冻结预算 | S5；`FORMAL_RUN.json`实际快照；完整1–3000epoch与train_end，四rank/global4096/seed42，launcher退出0；`FINAL_STAGE2.json:1f9e14b1595bac79feb378762604315e81708fdec245c1f9e53c851342e6ec24` | PASS |
| C2 | 四个完整有向ego token、全局身份、无测试特征 | S2；`assets/v10_direction_ego/audit.json:fff152cd88f18c71d05f06faff68bef2c6e43b311c62d5d7d91c94c47ca253dd`，两类各24587token、全部边，0mismatch；全目录GPU提取 | PASS |
| C3 | 每层三专家门控，共享一次GNN | S3/S4；`VALIDATION.json:90286cb217bcb0212a300467b823903691400b9934c51834529615aaf7744b21`与全部逐商品权重/attention | PASS |
| C4 | 实际SID累计硬前缀、相同抽样/参考 | S4/S5；两模型全部商品三级逐位一致；`evaluation/EXTRACTION.json:29eb0830d15d052a101fb005bd62a4be17f63b1bcd40bc5f02c3f2cea97f31fe` | PASS |
| C5 | 完整原损失梯度校准λ、α=1/3与冻结尺度 | S5；真实批次红→绿回归；实际初始化完整原梯度0.0011229729279875755，λ0.00008766930113779381；最终源未漂移 | PASS |
| C6 | 明示有偏训练、全量同口径独立评价 | E/X；6格各28391948三元组/24326中心、全原参考、统一尺度；`evaluation/FINAL_EVALUATION.json:6aa757d6453c92860b7e9a8072b0e8f029532c82904c190a26b5e6bfd786d1a7` | PASS |
| C7 | 复用不可变assets，只新建缺失方向ego | S2；`assets/manifest.json:cfbd9d9c0d3c5282f9fdd0c0b28798cbf7a86479c2626b702e499b589acc5b6b`，INFRA与重载测试；未重建旧资产 | PASS |
| C8 | 完整四卡profile、至少30连续inclusive epoch | 真实准入均值0.750110548s、正式窗口0.756988079s均<0.8；目标0.4–0.5未达；`PERFORMANCE.json:31059a38f2a1a183f675dd054aab7b25e8d668a837369e86e81df491402bf84d` | PASS（硬门；目标未达） |
| C9 | 真实数学/梯度/DDP/入口、反向与变异检查 | S2–S5；预派发104+61测试、18实际更新模块与四rank；后104+62及当前工具64通过；原预派发 `TEST_RESULTS.json:eca5d9bf5b60651dcc1de9763dc5f6c96177b9f2b7e3ee2b9cd32e451e398a19`；新工具 `POST_RECONCILIATION_TEST_RESULTS.json:7d0e1a7f66db347d177a33b5c1ba27cadce22e46713f8c5621508e0a59199e7f` | PASS |
| C10 | 一次合法Stage2后按碰撞门运行一次冻结Stage3 | 修正版Stage2完整；457/24587不同三码，collision0.9814129417985114；MCP实际拒绝、未豁免；`STAGE3_REFUSAL.json:b54a9e41a55412e73caa001c5c201e33fe267abe23d9ea98b696df1b7bf086ed` | BLOCKED |
| C11 | 前缀/行为/heldout/全量曲率/重建/逐商品分布 | E/X/R；全部记录已保存，负结果与宏/微分歧不隐藏；六格原文件与目录数组的精确路径/hash在完整评价中；`RECONCILIATION.json:d9757df2ce1d49dbaa6f8a79459a0bc6043268422eafeaedc413ee956bc6afe3` | PASS |
| C12 | 结构与同一57439总体Recall假设、配对、历史参照 | 结构选择性不支持，塌缩；#255的R@5/10、NDCG@5/10、配对均NA。#253/#252历史与>0.065阈值分开列明，不作单模块因果或未测Recall结论 | BLOCKED（推荐检验未执行） |
| C13 | 全量证据、独立复算、精确R2与推送 | 本地完整源/权重/SID/逐商品及逐中心数据保留，报告复核PASS；四件套缺Stage3 recall，正式归档器dry-run退出1，未部分上传；`ARCHIVE_STATUS.json` | BLOCKED（四件套不齐） |

汇总：**10PASS、3BLOCKED，0未修复工程FAIL**。不能把阻塞项改名为PASS；负面科学结果和未执行的推荐检验也不能互相替代。

## 七项机制与测试证据

1. 冻结数学/数据合同及论文对照：本合同§1–9及上述已冻结继承对照，未为指标放宽公式。
2. 独立参考：实际候选 `tests/reference.py`、`reference_layer_gate.py`、`test_reference.py`；ambient闭式与生产稳定径向/弦公式不同推导，极小图枚举与稀疏实现不同路径。
3. 手算答案：`tests/test_known_answers.py`、`test_layer_gate.py`、方向ego测试；不是把生产输出当期望值。
4. 12指定变异覆盖数值、mask/reduction与梯度路由；`MUTATIONS.md`、`MUTATIONS.json`、`LAYER_MUTATIONS.json`。崩溃或导入错误不计变异捕获。
5. 实际模块检查：`MODULE_CHECKS.json:d68b1d869ab888b17903f07ed80be055fa727a72a2ac0f00d6f8371cfe2495c2`，真实原/结构loss，未加权/λ/3加权梯度、重建夹角、clip及参数更新；不以非零梯度代替硬SID证据。
6. 实际四rank/入口/导出：原始准入和原始rank报告先于纠错代次，精确hash在派发凭据；正式源快照之后未改写，最终GPU代码重放全目录逐项一致。
7. 结论复核：R从原SID/原CSR与全部六份逐中心原始记录独立复算，退出0；`REPORT_REVIEW.json`再对本文报告的145个数值做原始SID、边、heldout银行、目录数组及时间边界核对，最大绝对差4.946941745131994e−10，小于声明1e−8。

修复实际工程故障：λ错误分子、空/四token旧fixture、验证根目录、历史manifest追加、规范SID数值策略、rawCSR归约。最后两个只改分析工具，未改训练快照、权重、SID、资产或正式训练；没有用修工具作为重训理由。归约器的原始CLI格式错误是真实复现，新helper尚未存在时的AttributeError不冒充指定变异捕获。

## 已排除解释与保留风险

| 解释 | 实测排除依据/边界 |
|---|---|
| 没跑满、用了旧代或无效低碰撞代 | 全部3000epoch、train_end、launcher0；r1源/权重/SID/archive_binding同代；无效代保留且排除 |
| SID不是checkpoint实际输出 | 新代最终冷重导出0差异、无重写，GPU目录提取前三码及第四计数器精确相等 |
| #253向量不是其规范SID | 原错误提取被拒；修复后highest全目录逐项一致；#255 high亦一致；共同float64几何口径不变 |
| 梯度断路/NaN/伪模块更新 | 原实际batch18模块与逐层路径/更新/DDP证明；正式有限loss与最终代码/输出核验。此证据不声称排除了所有可能建模缺陷 |
| 大幅共享率就是结构收益 | degree-matched对照更高；L1最大桶23293/24587、三码457种。原始共享提升不可解释为选择性收益 |
| R2凭证缺失或上传失败 | 认证读操作退出0；真正缺口是被硬门阻止的Stage3 test_final，正式归档器据此退出1，未调用部分上传 |
| micro改善证明机制成功或Recall改善 | macro相反、SID门失败、没有Stage3新test；仅单次组合机制观察，无单模块因果、泛化排序或噪声归因 |

低quant_error0.009193088与effective_rank2.199264266只是带bias encoder/硬码诊断，不改变唯一碰撞硬门。训练侧、结构held-out和未执行的推荐test分开；全量评估不把有偏训练变成无偏训练。

## 阻塞与最小owner决定

失败命令/原文：`stage3_start({item_sids: <r1>/result/item_sids.json})`返回 `started:false`、`collision_gate_waived:false`，原因 `SID quality gate: three-level collision 0.981413 is not below 0.40`。`python3 results/research/issue139/upload_stage3.py TIGER_prefix_curvature_gate_rq --dry-run`退出1：`ValueError: missing necessary Stage3 test_final.json for TIGER_prefix_curvature_gate_rq`。

- A（本次遵守）：保留 `<0.4` 与单次正式预算，接受此Stage2负结果，原合同闭环为BLOCKED，不伪造Recall或部分R2。
- B：owner明确修改入场合同，允许一次标明 `collision_gate_waived` 的out-of-gate Stage3，才能取得缺失Recall并补齐四件套；这不是当前已有授权，也不能冒充原冻结合同PASS。
- C：另立新机制合同/预算，不能对本臂追分调参或把新实验冒充原单次结果。本用户已要求随后执行#256，但本issue关闭前未提前读取/开展它。

## 关闭检查

1. 生产机制与全部可执行分析代码是否实现：**是**，实际源与E/X/R均完整、无stub。
2. 必要且可执行的测试是否实际执行：**是**，104候选、当前完整工具64、实际四rank/入口及全量六格/独立归约均有真实输出与退出码。
3. 全部合同训练是否完成：**否**——Stage2是3000epoch；Stage3被实际硬门拒绝。因此禁止COMPLETED关闭，只适用§6.1的冻结规则冲突BLOCKED关闭。
4. Infra是否来自assets且未重复创建：**是**，仅缺失方向ego新建于v10，原资产按不可变hash复用；详细来源在INFRA/manifest。
5. 在冻结规则内是否仍有可执行的未完成科学/工程任务：**否**——可执行评价/修复/复核均完成；剩余科学产物必须先获得owner的合同变更。交付的选择性提交、推送SHA核对与issue关闭状态确认必须执行，精确交付commit以issue关闭评论为准。

这是有证据的BLOCKED结案，不是“保存断点后停止”；关闭并确认CLOSED后才按用户顺序开始#256。
