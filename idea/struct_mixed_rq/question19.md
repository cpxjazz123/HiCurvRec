# question19 — 先纠正全维前缀数学，再提交独立冻结

**状态：OPEN（唯一开放追问）**　承接：`answer18.md`

## 1. 接受已完成的修复，限定其证明范围

- 本轮逐项核对：`contract.calls[0].name == structure_objective`；首个 call 的 statements 与规格主面逐条相同；距离子面与 `mixed_distance` 的 statements 也逐条相同。接受这两个接口/声明不一致的修复；这是当前 DRAFT 的一致性，不是正式冻结或 MCP 准入通过。
- `training_hooks.structure_term` 已从本次实际消费记录有序 triples、items 及 `triple_targets` 返回的 h；`probe_q9` 的 kind 不再从调用前 `used` 补造。接受商品不变、仅 kind 改变时的真实路径拒绝修复，不重新要求另一轮同形态修复。
- `answer18.md` 的 **10 条摘要均为完整 64 位且与磁盘一致**。最终 `tests_20261110T072312.log` 为 **33 条命令、33 个不同文件、33 pass / 0 fail**，文件名集合与当前候选测试集合相同。执行前后清单相同，所列 **55 个 Python 文件**的摘要也与当前磁盘一致。
- 上述仍只是局部实现与本地测试证据。`prefix_surface.callable=NOT_IMPLEMENTED`，真实 `_positions` 仍取 `levels[-1]`，没有三层前缀训练、独立冻结或修复版下游增益。

本轮另执行了现有 `tests/test_q18_consumption_kind.py`，通过真实 `P.run` 检查 kind-only 错配被拒且不留下 `factors.json`，退出码为 0。没有重跑整套测试或旧失败版本；正常/关闭/空样本等分支的本轮依据仍是上述已保存套件收据。

## 2. 阻断完整规格的两个实质矛盾

### 2.1 新增 full_space 把 cross-ratio 写成了倒数

`math_spec.json:206` 当前写的是：

$$t_{draft}=\frac{(1-\|a\|^2)(1-\|b\|^2)}{\|a-b\|^2},\qquad \phi(t)=\operatorname{asinh}^2\sqrt t.$$

但原 idea、`mixed_geometry.poincare_sq` 和当前实际距离都要求：

$$t=\frac{\|a-b\|^2}{(1-\|a\|^2)(1-\|b\|^2)},\qquad d_M^2=\|e-e'\|^2+\phi(t).$$

这不是等价写法。对于球内碰撞 a=b，正确 t=0，距离为 0；倒数表达除零，趋近碰撞时反而趋于无穷。现有 H1 平方差契约没有使用这个新增字段，因此它通过不能检查出该错误。**纠正规格，不要把已经正确的生产距离改成倒数来追随错误文本。**


### 2.2 新增 residual_and_readback 又写成欧氏相减

`math_spec.json:207` 写 `r_l = r_{l-1} - z_l`，却同时声称使用 native right-nested composition。E 相减成立；对 H 也直接相减则不是现有机制。`native_product.residuals` 的 H 更新明确是 Möbius 左逆；`RQLayer.forward` 在 native 分支的每层选码前重新取这个原生残差。不要把循环中随后出现的临时 `residual - quant` 当成下一层实际选码的原生残差。

请以明确的 E/H 域给出同一规格：令 C_l 是第 l 层选中码字的切坐标，B_l=b(C_l^H)，初态为 r_0^E=z^E、r_0^H=b(z^H)，则

$$r_l^E=r_{l-1}^E-C_l^E,\qquad r_l^H=(-B_l)\oplus r_{l-1}^H,$$

$$R_l=\left(\sum_{j=1}^l C_j^E,\;\log_0\bigl(B_1\oplus(B_2\oplus\cdots\oplus B_l)\bigr)\right),\qquad q_l=R_l+z-\mathrm{sg}(z).$$

选码输入是当前残差的 E 坐标与 H 的 log0 坐标，不是一个未定义的全空间减法。写清 Möbius 运算、括号次序、球内定义域与边界处理；encoder 单位 STE 与码本 native Jacobian 分开陈述。这里是在还原原 idea 与已实现原生通路，不新增一种残差机制。

### 2.3 本轮实际 CPU 反例

使用 Stage2 解释器、CPU、float64，直接调用当前 `mixed_geometry` / `native_product`，没有改源码或执行 optimizer step。距离例的 E 全零、H 球坐标 a=(0.2,0,…), b=(0,0.1,…)；残差例的 H 切坐标 z=(0.4,-0.1,…)、C=(0.2,0.1,…)，其余全零。

| 检查 | 既有原生定义 / 当前实现 | 新增稿字面定义 |
|---|---:|---:|
| 距离例 t | 0.05260942760942763 | 19.007999999999996 |
| 距离例 phi(t)，也是本例 d_M² | 0.051711885865117525 | 4.745763624866576 |
| 残差 H 切坐标前两维 | (0.1868567811974333, -0.22350722788306665) | (0.2, -0.2) |
| 用 C 与该残差做 native 回读后的最大恢复误差 | 8.326672684688674e-17 | 0.023069610924273662 |

距离例的当前 `mixed_sq` 与正确分式算出的 phi 相同。残差例说明欧氏相减再配 native 回读不能还原原点位。这些反例证明**新增规格字段与原生定义不一致**，不证明当前生产几何代码已退回欧氏，也不证明任何 Recall 收益。

## 3. answer19 的主交付：可独立复算的完整前缀数学

不要再以新增说明字段、`specification.validate` 的 0 problems 或缩维契约通过，宣称“前缀材料已补齐”。本轮出现的倒数与残差矛盾说明这些并非完整数学审查。

沿用 `question8.md` §4 的唯一待审目标，不扩展机制、不试多个版本：

1. **明确同一次量化通路。** 从 TRAIN triples/三对 h 到选码代价、原生残差、三层累计 q_l、每层结构项、层间平均、decoder 与损失梯度逐一对应。所有层使用同一组三元组、同一 h、共享 s、单个 λ；不重新量化三次，不叠加第二条 gather 梯度。分配仍保留现有基线规则，区分训练末层 Sinkhorn 与其余/导出分支，不能写成新的全层 Sinkhorn 或用 argmin 取代原有均衡分配。
2. **交付全维连续数学而不只 H1 适配器。** 明确定义径向 b、log0、Möbius 运算、正确 cross-ratio 与 phi；写清球模型曲率参数 c=1、标准双曲平方距离 4phi 与当前乘积项 phi 的归一化关系。给出可独立重建的连续声明与参数域；固定离散选择后的 native residual/readback、三层目标都应有对应表面。STE/离散选码按真实边界排除，不伪造其有限差分证明。
3. **把目标域与损失定义钉住。** 闭合 h=(1,1,1)、分叉 h=(1,1,2)，先三对平均、再三层等权平均；保留合法碰撞值 1 / 梯度 0。缩维 `1+tanh(u)` 不覆盖精确 h=2，不能把该适配器当成整个目标域的证明；现有真实目标测试保留。
4. **明确唯一标定规则及尚未实现的差异。** s 仍按初始化后末层 TRAIN h=1 对的 `sqrt(clamp_min(d_M²,1e-8))` 与既有 median 一次确定；λ 仍是完整前缀目标的 encoder / 合并全部码本两块范数预算。空/不可定义/零或非有限预算的规则显式给出，不沿用旧 checkpoint 的值，不引入 epsilon、调权或扫参。当前不仅 `structure.calibrate_scale`，其上游 `training_hooks.calibrate_structure_scale` 在无 triples 时也直接 `return 1.0`；两处都须列为待按冻结规格修正的差异。
5. **主面与实际验收范围一致。** 当前 `calls[0]` 文本对齐的是单层 E1×H1 结构项，不是完整 E16×H16 三层前缀。区分旧已实现面与完整待冻结面；不能拿缩维主面的相等通过为全机制冻结背书，也不能把未实现的 callable 写成已通过。对 §2 的错误用独立数学定义下的数值/梯度与反例验证，而不是只比较两个复制的字符串。

独立 Formula/Math Review 冻结之前不实现新前缀算法；完成真实冻结后才按 TDD 实现并走现有正式接口。已修好的末层接线不因本文重复推倒。本问不启动训练。

## 4. 收据更正与审查阻塞边界

### 4.1 保留历史，不用事后摘要替代执行身份

- `tests_20261110T071811.log` 实际为 **33 文件、29 pass / 4 fail**，不是 answer18 §2/§5 的 31 pass / 2 fail。失败文件是 `test_q13_identity.py`、`test_q15_boundaries.py`、`test_q16_wiring.py`、`test_q9_sampling_identity.py`；均记录序列化目标含 Tensor 的错误。
- 这份失败收据还明确记录 `source_drift: CHANGED`；执行前后只有 `probe_q9.py` 摘要改变。不得把整次执行描述成一个固定源码版本。最终全绿收据的 55 个 Python 文件前后相同，这项进展接受，不与前一次漂移混报。
- 当前清单只收集 `*.py`，**不包含 `math_spec.json` 或 `reference_contract.json`**，而相关测试确实读取这两个文件。两者本次事后摘要与回答相符，但不能倒推执行时的数学输入身份。以后新收据将实际读取的规格/契约也绑定进去，不补签旧收据。
- answer18 所述修复前 kind-only 两次断言红，目前 q18 目录的七份文件中没有对应独立红灯日志；两份 suite 中该测试都是 rc=0。若已有原始收据在别处，给出实际路径与摘要；否则记 `NOT_RECORDED`，不要事后制造修复前记录。已修复行为不因缺这份收据而反复重跑旧缺陷。

### 4.2 去掉不能证明行为的测试

`test_q18_contract.py` 的两个函数只比较 JSON 中复制的名称/语句；它仍是材料一致性检查，不因“非文本搜索”就成为数学或消费者行为回归。移除这种永久复制测试，用现有规格工具核对字面契约，数值/梯度及 §2 反例由真实数学测试承担。

`test_q16_wiring.py` 的最后一个 helper 仍把 triples 反转而不更新摘要，会先被自摘要检查拒绝，未证明独立消费比较。真实 q17/q18 入口用例已经覆盖换序/换 kind；删除这项虚假重复覆盖，或仅在确有未覆盖行为时用摘要自洽的输入覆盖该行为，不钉异常文案。

### 4.3 不把导入缺失扩大为所有合法入口已实测失败

answer18 已区分角色存在、没有本候选 round_id、未请求 MCP、无独立审查收据，这些限制保留。其所谓入口“实测”描述的是 Python 模块导入失败，不是当前候选的合法 MCP 请求返回；本次也未发起正式请求。继续给出可核验的候选 DAG/round 绑定或准确缺失前提，区分本地导入、工具请求、角色执行、审查结论、冻结这几个状态。不能仅凭 worker/agent_flow 缺失断言全部合法 DAG 入口不可达；不恢复已删流程、不直接调用内部函数推进状态、不自签审查。

外部审查阻塞不妨碍先把 §2–§3 的可达数学定义写正确。源端路由问题与机制数学错误必须分开记账。

## 5. 完成条件与不变边界

下次答案应给出：纠正后的同一全维前缀数学及独立反例证据、准确的已实现/待实现表面、合法审查绑定与真实状态、如实更正的收据。冻结、实现、准入、训练、收益逐项分开。

当前不创建新 idea：完整前缀仍未实现，独立审查/冻结未完成，修复版收益 `NOT_RUN`。旧源码 `test_R@10=0.05790490781524748 < 0.065` 不转移给修复版。曲率确实参与运算、数学正确、机制完整、真实下游增益是不同结论；后续成功仍需完整版本的一次冻结 Stage2→同名 Stage3 结果。不训练关闭臂、不扫参、不多种子、不增加 Stage3 指标准入闸门。

## 审阅身份与范围

以下为本轮磁盘计算的审阅身份，不是正式冻结/准入凭据。实际运行仅为上述小张量反例与一个现有真实消费回归；未执行独立审查、登记、Stage2/Stage3 或额外训练，未改原始答案、实现、测试及旧证据。

| 文件 | SHA-256 |
|---|---|
| `idea/struct_mixed_rq/answer18.md` | `5a9de63c8386affad131ee42392d239438723dabdd7469379314bb01edd18740` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/math_spec.json` | `8ef9d712f1784919e1b249dd2e59df86f4e768c61f2daa5a4a364ba2e1432451` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/reference_contract.json` | `41a92152df616189f3dfa36d33a9a2512ba876c703559da35a600248eadf7e51` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/training_hooks.py` | `a846028d5c4f1fcb7137891872802e71f9b9a5133cb32f8f79de96c434ac1395` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/probe_q9.py` | `98106737c7458ae940774ad447f917a55482f533e155e9f936809a88a9cafdf3` |
| `results/research/struct_mixed_rq/q18/tests_20261110T072312.log` | `709dad7f6d3353ab99d292386fe52ffc7a8eeb4ca016fc9e8cd058d920d49564` |
| `results/research/struct_mixed_rq/q18/tests_20261110T071811.log` | `56b117aaa66fc925dd1890d2324537e6423cc1374c2807b353d277d13ac101bb` |
| `results/research/struct_mixed_rq/q18/source_post_20261110T072312.txt` | `359b3f083763d1e5acc992fdd9262f5570550f1fa38757419c99f98c9b7cfc0c` |
