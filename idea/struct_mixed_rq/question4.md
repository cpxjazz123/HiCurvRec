# question4 — 原生回读的 STE 是否把梯度放大、反向或抵消？

**状态：已审阅归档（缺陷已实测确认，尚未修复；转交 `question5.md`）**　提出：2026-11-10　承接：`answer3.md`　审阅：2026-11-10

> 主线不变：曲率真正参与 RQ-VAE 并带来下游增益。先让实际梯度与机制声明一致，再讨论收益；不以新损失、逐层监督或调大 λ 掩盖接线问题，不重训或改变现有实验。

## 1. answer3 的进展与不能关闭的部分

本次确实找到 `results/research/struct_mixed_rq/q3/diagnostics.json` 和脚本副本；副本 SHA 与原脚本、JSON 里的 script SHA 一致。保留真实记录：采用正式标定值的 checkpoint 检查中，encoder、C1、C2、C3 的加权结构梯度范数均非零，decoder 未连接；train/export 的 L3 差异仍为 42/64，同模式 geometry-on/off token 差仍为 0/0/0。本次未重跑这组 TRAIN 测量。

**不接受“契约与逐层证据齐备”的无保留 CLOSED。** 核心缺口不是又少一张范数表，而是回读的代理梯度公式仍写错，并漏掉了可能抵消码本梯度的通路。

- `answer3` 把 `c_l` 称为“真实 gather，不带梯度”。离散索引不可微，但 gather 对选中的码本权重可微，二者必须区分。
- 文中 on 公式使用 `sg(sum sg(c_l-r_l))`，却说它提供编码器梯度。外层 sg 的梯度就是零，而且该项的前向值通常不为零。
- 当前源码用的却是 **`T - sg(T)`**：前向为零、反向保留 T 的梯度。这与报告公式不是等价改写。
- nonzero/finite 的逐层合并范数只证明存在梯度，不能证明每个几何因子的方向、倍率与声明一致。

## 2. 按当前代码推导真正的代理 Jacobian

权威源码：`model/layers.py::VQLayer.forward` 第 142–147 行、`RQLayer.forward` 第 305–328 行，以及 `native_product.py::residuals/reconstruct`。

固定离散选择，记 `R(c)` 为三个码字的原生回读，`sg` 为 stop-gradient：

\[
r_l=\mathrm{NativeResidual}_l(z,c_{<l}),\quad
u_l=r_l+\operatorname{sg}(c_l-r_l),\quad
T=\sum_{l=1}^{3}u_l,
\]
\[
q_{\rm current}=R(c_1,c_2,c_3)+T-\operatorname{sg}(T).
\]

T 并非只依赖 encoder：native 残差每层都从 z 和前层**可微码字**重新构建。忽略这条间接路径，就会误把额外梯度当成“一次直通”。

### 精确 E16 因子，不需要令曲率趋近于零

E16×H16 的 E 半区本来就是欧氏。其三层残差严格为

\[
r_1^E=z^E,\quad r_2^E=z^E-c_1^E,\quad
r_3^E=z^E-c_1^E-c_2^E,\quad
R_E=c_1^E+c_2^E+c_3^E.
\]

**[INFERENCE / 源码与 stop-gradient 语义推导，尚非候选运行实测]** 因而当前图的局部 autograd/STE 代理 Jacobian 为：

| 对哪个变量 | 原生回读 R 本身 | T 的代理贡献 | 当前 q 的合成贡献 |
|---|---|---|---|
| z 的 E 半区 | 0 | 3I | **3I** |
| C1 选中码字的 E 半区 | I | −2I | **−I** |
| C2 选中码字的 E 半区 | I | −I | **0** |
| C3 选中码字的 E 半区 | I | 0 | **I** |

这里讨论的是 **STE 指定的代理反向**，不是“硬 argmin 前向对 z 的经典导数”。不能用整条 STE 的有限差分 gradcheck 证明代理规则正确。

这解释了为什么“C2 的 32 维总体范数非零”仍不足以验收：H 半区的非零分量可能掩盖 E 半区抵消。它是否解释该 checkpoint 的具体范数，仍须逐因子真实检查；不能由这张解析表直接宣布真实数据上的因果结论。量化损失还有独立的码本路径，不应称 C2 完全不训练。

### 本次检查的诚实边界

已独立执行纯标量仿射/stop-gradient 代数核对：取 `z_E=0.8`、三个选中码字 `0.1/0.2/0.3`，当前源码方程的值为 `0.6`，代理系数为 `(3,−1,0,1)`；`answer3` 印出的方程值为 `−0.8`，代理系数为 `(0,1,1,1)`。它验证的是上述方程不等价，不冒充真实模型执行。

另尝试真实 `RQLayer` CPU float64 检查：geometry-on、eval、3×256 码、32 维，H 全零，三个码本第 0 个码的 E 首坐标为 `0.1/0.2/0.3`，其余码为远点 `7+k/1000`，输入 E 首坐标 `0.8`，读取 q 的 E 首坐标对 z 和三个选中权重的 autograd。该尝试 **120 秒超时，无返回结果，记 TIMEOUT**；不能称生产 Jacobian 已实测、测试已红或已绿。没有 optimizer step、GPU 作业或 checkpoint 写入。

## 3. 下一步必须回答什么？

### A. 先钉住目标梯度，再动实现

1. 更正 `answer3` 的 on 方程、gather 与 sg 描述；原报告作为历史证据保留，另写勘误，不倒改既有结果。
2. 给出 encoder 和每层码本应有的代理 Jacobian。主说明现有声明是“encoder 切空间直通，码本 native 真梯度”，其通常含义是 encoder 一次 I、码本为 R 的真实连续 Jacobian；必须明确采用或说明冲突，不能用“只要非零”替代。
3. **仅当该目标经规格确认后**，可用
   \[
   q_{\rm ref}=R(c_1,c_2,c_3)+z-\operatorname{sg}(z)
   \]
   作为固定选码下的参考：前向仍为 R，encoder 代理为 I，码本不额外承受 T 的间接梯度。这是待规格确认的参考，不是本问已批准的实现更改；还须逐层定义累计回读与残差，不能只改末层掩盖前层。
4. 当前工作树 `math_spec.json` 是 DRAFT，旧运行的冻结规格/run_id 仍 UNAVAILABLE。先按现有规格与审查流程明确新版本；不得把新定义倒填成旧 checkpoint 的训练契约，也不得把不可得的历史身份编造出来。

### B. 用能抓住方向与倍率的测试，不只检查非零

- 对实际 `RQLayer` 写可执行的解析 E 因子回归，覆盖各层累计输出及最终输出。区分前向正确和反向错误；按确认后的目标检查 encoder 倍率、C1 符号、C2 是否抵消、C3 路径。当前代码必须出现真实可解释的红灯，不能以缺模块/超时充当行为反例。
- 连续 R 在固定码字上的 H 因子取值与梯度，使用独立参考核对；STE 与离散分配仍列为不可由普通 gradcheck 覆盖的部分。
- 真实 TRAIN batch 的重构项、量化项、结构项分开检查 encoder、C1/C2/C3 的 E/H 梯度及方向、None/zero；完整 32 维范数非零不能掩盖半区反向或抵消。若规格要求参数更新检查，只在廉价 both-on 接线测试内完成，不训练关闭臂。
- `test_rqlayer_native_path.py` 目前只要求梯度非零；`test_structure_positions_gradient.py` 比较 hook 与同一个 `levels[-1]` 图，能防额外叠加，却不能抓住共有的 STE 错误。需补解析反例，不能再次用同一路径自证。
- 修复必须先真实红灯、再改隔离源码、最后按仓库要求验证。**本问不直接修改当前运行源码，不重跑正式 Stage2/Stage3，不覆盖旧 checkpoint、SID 或已冻结证据。**

### C. 将现有证据补齐到可复核，不伪造新成功

`answer3` 的数据归档已有实质进展，保留；以下缺口随同本问补齐，不另开并行问题：

- 0.0742 是 encoder 的 **structure-vs-total** 余弦，不是此前要求的 structure-vs-base。补 `base=recon+quant` 的逐块角度；按保存的标量推算，当前 encoder 后者约 0.004254（仅代数推算，不代替梯度重算）。不能把两种参照混为一项。
- 原始 decoder 余弦为 `1.0001065731`；用高精度累积检查数值误差，保留原值，不靠 clamp 隐藏问题。
- 脚本的 s/λ 仍是与日志相等的硬编码，而非可追溯读取；补实际标定日志的路径与 SHA。补采样三元组、执行命令、stdout/stderr、测试摘要及红→绿收据；当前 q3 证据目录只有 JSON 和脚本副本。
- 几何比较中 token 用 export 模式，累计回读另在恢复后的 train 模式计算。分别标明 mode，不能将两个统计描述为同一次同模式前向；若比较连续量与硬码的关系，须固定同一前向与模式。
- 仍无法恢复的历史材料标 UNAVAILABLE，不把后来的诊断称为原运行复现。

## 4. 完成标准与下游边界

本问关闭需要：正确的前向与代理梯度定义；真实生产 Jacobian/解析回归证据；必要修复的 TDD 与验证记录，或明确说明为什么无需修复；可复核的实际执行材料。不得以非零范数、TIMEOUT 或文档自称 CLOSED 代替。

本次读取的 Stage3 日志最新 train 事件为 **125/150**，尚未发现最终 `test_R@10`。这不是存活检查。本问不启动、不取消、不等候该运行；其最终结果属于**旧源码图**，即使随后超过 0.065，也不能移植成修复版本的收益证据。若原流程完成，按既有冻结协议、同名配对、SID SHA 和 R2 规则记录完整结果。

逐层前缀监督仍未实现；不能用新增该损失绕过上述基础梯度问题。完整机制与最终收益证据未齐，**不创建新 idea**。

## 5. 本次审阅绑定

| 文件 | SHA-256 |
|---|---|
| `answer3.md` | `e7b3099065980fef7681cfe8ce2fc4cbdabe5dc6505759267966d459a3cf9aa1` |
| `results/research/struct_mixed_rq/q3/diagnostics.json` | `f9557408f5e96d2a6d38a99f50e2f208a5ebd99039c9183d93be77679f71ccf0` |
| `precheck_question3.py`（原件及归档副本） | `fc356f333deabc9ea125bb52df847273ca8a1f82c549c550b30518b00a56e5fb` |
| `model/layers.py` | `51b287f6e280b09a5d3813d3b0050dda94b2cbec8edcda4eb1538b4a5b4f95d3` |
| `native_product.py` | `3853f061eb82e2ed0c51236308df094308c31a4b98d2dfcde5a55a6b5a03cc25` |

## 6. answer4 审阅结果

已找到固定选码的真实 RQLayer 输出 `(3,−1,0,1)`，原脚本与归档副本一致；前向正确、代理梯度与声明不符的缺陷确认。但脚本只打印实际值和目标值，没有失败断言，不能将其当成已执行的 TDD 红灯；实现也未修改，因此不接受修复完成或无保留 CLOSED。

另外，q4 梯度表取自 `grads(term)`，是未乘 λ 的范数；乘正式 λ 后 encoder 结构范数约 `1.41558e-4`，不是 153.4338。q3 的 decoder `1.0001065731` 是 recon-vs-total，不能靠换测 structure-vs-base 为 UNDEFINED 宣称原精度问题已复核。勘误和算术见 `question5.md`，原始 answer 不改写。

最终 `test_recall@10=0.05790490781524748 < 0.065` 已由 `test_final.json` 核对；该旧源码实验未达成功条件，也不能成为未来修复版本的收益证据。唯一开放追问为 [`question5.md`](question5.md)，明确要求在隔离版本冻结目标梯度、写真实失败测试、实施修复并验证，而不是再次停在诊断建议。
