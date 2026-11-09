# struct_mixed_rq — 闭合/分叉结构监督下的固定 E16×H16 乘积空间 RQ-VAE

> 主线（不变）：把曲率几何真正接入 RQ-VAE 的选码与重构通路，并以冻结协议下的 Stage3 `test_R@10` 判断是否获得增益。
> 候选树：`stage2_RQ-VAE/TIGER_struct_mixed_rq`。本文是机制说明，不是准入声明。
> 本目录的开放问题按 `question1.md`、`question2.md`… 顺序推进；**同一时刻只允许一个未解决的问题**，前一个关闭后才建下一个。

---

## 1. 问题

Stage3 只读离散 token。所以"曲率有没有用"必须落在**硬 SID 的前缀分组**上，而不是连续空间里的距离或 loss 数值。

| 实测事实 | 数值 |
|---|---|
| 基线 L1 行为边保留率 | 18.4% |
| 行为最优同尺寸分组（Louvain≈167/411 组） | 20–25% |
| 低度边（两端度≤10）与最优分组的差 | 35–40pp |
| 训练行为图 | 24,474 节点 / 264,326 边；46% 边不在任何三角形内；top 1% 节点持有 60.6% 的三角形；聚类均值 0.0390（同度重连 0.0232） |

探索性相关（n=55，用户提供文本，未独立复核，仅作动机）：组内平均局部聚类系数与 Recall 的 Spearman 在 L1/L2/L3 为 0.538/0.616/0.500；AF3 组内外 Gap 为 0.477/0.480/0.464。相关不是因果，且 AF3 与端点度数混杂：

$$\mathrm{Gap}_{AF3}=\bigl(\mathbb E[d_u+d_v\mid\text{外}]-\mathbb E[d_u+d_v\mid\text{内}]\bigr)+3\bigl(\mathbb E[m_e\mid\text{内}]-\mathbb E[m_e\mid\text{外}]\bigr)$$

**本 idea 不追求"把 AF3 拟合得更准"，而是让闭合与分叉两类关系在量化后都得到正确表达。**

---

## 2. 几何：固定 E16×H16 乘积空间

- 32 维潜在 = 前 16 维欧氏（E）+ 后 16 维双曲（H）。
- H 半区按**切向量图**解释，经径向映射进入曲率 1 的 Poincaré 球：

$$b(h)=h\cdot\frac{\tanh\lVert h\rVert}{\lVert h\rVert},\qquad \lVert b(h)\rVert=\tanh\lVert h\rVert<1$$

- 乘积平方距离，H 部分用稳定等价式（`phi` 形式，避免 `acosh` 在零点附近被截断）：

$$d_M^2=\lVert e_1-e_2\rVert^2+\phi(t),\qquad \phi(t)=\operatorname{asinh}^2\!\sqrt{t},\qquad t=\frac{\lVert a-b\rVert^2}{(1-\lVert a\rVert^2)(1-\lVert b\rVert^2)}$$

- `phi(0)=0`、`phi'(0)=1`；小 $t$ 用级数 $t-\frac{t^2}{3}+\frac{8t^3}{45}$（$t<10^{-3}$，截断 $O(t^4)=10^{-12}$）。
- 只在数值饱和（$\lVert b\rVert$ 舍入到 1）时把点拉回球内；普通点不收缩，`log0(b(h))=h` 严格成立。

**这不是位置依赖曲率的异质流形。** 固定乘积空间只提供"能容纳多分支"的归纳偏置，不声称曲率随节点变化。

---

## 3. 量化：原生残差 + 右嵌套回读

保持基线协议：3 层 × 256 码、最后一层 Sinkhorn 均衡分配、`CODEBOOK_DIM=32`、全局唯一 SID。改动只在该协议**内部**的几何运算：

| 环节 | 基线（切空间） | 本 idea（原生乘积） |
|---|---|---|
| 残差 | $r_{l+1}=r_l-c_l$ | E 相减；H 用 Möbius 左逆 $r_{l+1}=(-c_l)\oplus r_l$ |
| 重构 | $q=c_1+c_2+c_3$ | 右嵌套 $q_H=c_1\oplus(c_2\oplus c_3)$，decoder 读 $\log_0(q_H)$ |
| 选码代价 | 欧氏距离 | 同一 $d_M^2$ |
| 分配规则 | Sinkhorn（仅末层） | 不变 |

自洽性：$(-c_2)\oplus\bigl((-c_1)\oplus b\bigr)$ 对应 $b=c_1\oplus(c_2\oplus c_3)$。

**梯度边界（STE，必须在规格中显式声明）**

| 量 | 值 | 梯度 |
|---|---|---|
| 重构 $q$ | 原生回读 | codebook 走 native 真梯度；encoder 走切空间直通 |
| 结构项位置 | `levels[-1]` 单一路径 | 同上；**不得**再叠加 `codes_to_positions`（会双计码本梯度，已修） |

不变量：`to_ball(decoder 输入 H 半区) == c1⊕(c2⊕c3)`（允许饱和尾巴误差）。

---

## 4. 结构监督：闭合 vs 分叉

只用 TRAIN 合法可见行为构图（相邻历史对，去重、无向）。

| 结构 | 观测模式 | 三对目标 $h$ |
|---|---|---|
| 闭合三角形 | A–B、A–C、B–C 均有边 | (1, 1, 1) |
| 开放楔形 | A–B、A–C 有边，B–C 无观测边 | (1, 1, 2) |

"无观测边"只要求更远，**不当作真实负反馈**。

$$\mathcal L_{\text{struct}}=\mathbb E_T\Bigl[\tfrac13\sum_{a<b}\Bigl(\frac{d_M^2(q_a,q_b)}{s^2h_{ab}^2}-1\Bigr)^2\Bigr]$$

- 用**平方距离**：$d/\cdot$ 形式在 $q_a=q_b$ 处有尖点；平方形式在对角线梯度恰为 0（已用解析式 $4x/a^2(\lVert x\rVert^2/a^2-1)$ 与 `torch.equal` 双回归）。
- 诚实边界：碰撞处值 $=1$、梯度 $=0$，**不会自动拆开共享码字**；不另加未批准的防坍缩项。
- 采样：中心在全 catalogue 上均匀抽取，每中心限少数邻居对，防止枢纽组合数支配梯度。请求 4096 个三元组时实际接受 3024 个（无可用邻居对即丢弃），接受率必须进记录。

### 4.1 逐层前缀监督（设计，未实现）

同一方程施加于每层累计位置 $c_1$、$c_1\oplus c_2$、$c_1\oplus(c_2\oplus c_3)$，共用同一 $s$ 与 $\lambda$，不按层调权。理由：T5 逐 token 生成，L1 先行且容量最紧；只监督末层等于把约束全压在最细粒度上。

---

## 5. 标定（一次性、冻结）

| 量 | 规则 | 实测（batch 1024 / 4096 三元组） |
|---|---|---|
| $s$ | TRAIN 上 $h=1$ 对的中位距离，码本初始化后测一次 | 0.041920 |
| $\lambda$ | $\lambda^*=\min_b\lVert g_b^{\text{base}}\rVert/\lVert g_b^{\text{struct}}\rVert$，块={encoder, 全部可训练码本}，decoder 不计入 | 3.5211e-06（codebooks 定格） |

| 块 | base 范数 | structure 范数 | 比值 | 余弦 |
|---|---|---|---|---|
| encoder | 4.4649e-04 | 7.1867e+01 | 6.213e-06 | 0.1175 |
| codebooks | 3.4708e-05 | 9.8571e+00 | 3.521e-06 | 0.0708 |

必须如实记录的解读：

- 余弦 0.07–0.12 ≈ 近正交 ⇒ "范数相等"不是"10% 扰动"，而是一次约 30° 的转向；$\lambda$ 是否合适只能由实测 token 变化判定。
- 决定 $\lambda$ 的是 codebooks 块，其 base 范数比 encoder 小 13 倍（KMeans 后 commitment 近饱和）⇒ $\lambda$ 相对一个几乎为零的量定格，不能当效果证据。
- 退化即 `CALIBRATION_DEGENERATE`：不训练、不 `optimizer.step`、不导出。

---

## 6. 三个开关（只做廉价接线检查，不训练关闭臂）

| 开关 | 语义 |
|---|---|
| behaviour-off | 不建图、不算结构项；只剩重构 + 量化损失 |
| geometry-off | 同容量欧氏：选码代价、残差、重构**同时**走切空间，与基线欧氏实现逐元素相等 |
| both-on | 结构项 + 乘积测地距离（唯一训练的正式臂） |

单一真值源：`mixed_geometry.GEOMETRY_ON` 由 trainer 设置一次；`MIXED_ASSIGNMENT` 已删除。

---

## 7. 判据

**参与性（必须先成立）**

1. 曲线距离进入真实选码代价；Möbius 残差/合成/log0 回读进入 `RQLayer`/decoder。
2. 真实 TRAIN batch：仅重构 → encoder 与三级码本均非零梯度；结构项 → 两者均非零且一步 AdamW 后参数确实变化。还须核对 E/H 因子的方向与倍率，不能只凭完整张量范数非零验收代理梯度。
3. CPU 预检记录换码计数（见 `question1.md`）；联合损失训练前后的换码只证明训练轨迹在动，不能单独证明结构项或曲率的贡献。历次测量审阅见 `question2.md`、`question3.md`，当前原生回读梯度契约见 `question4.md`。
4. 记录对 token 重标号不敏感的指标：L1/L2/L3 同前缀邻居对比例、前缀数、L3 碰撞率、训练分支（Sinkhorn+argmax）与导出分支（argmin）的 token 差异率。

**最终判据**：一次冻结 Stage2（3000 epoch、全局 batch 4096、seed 42、无早停）→ 同名目录冻结 Stage3（150 epoch）→ `test_R@10 > 0.065`。AF3 Gap、聚类系数、距离失真只作诊断。

---

## 8. 明确不做

- 不做消融训练臂；不做调参扫描；不做多种子。
- 不用普通 argmin 取代 Sinkhorn 均衡分配。
- 不把"曲率拟合更好"当作"推荐更好"。
- 未过机器准入前不 `torchrun`、不启动 GPU。

---

## 9. 当前状态（2026-11-10 审阅 answer3）

已实现记录：固定 E16×H16 与 `phi` 距离；原生残差 + 右嵌套回读接入 `RQLayer`；结构项已去除旧的额外 gather 路径，但原生回读中的 STE 代理梯度仍未验收；单一几何开关；块预算标定（历史 CPU 读数 $\lambda=3.5211\text{e-}06$，不是正式训练的实际标定值）；旧 ratio 校准路径清零。

关键提交：`dd9f47cb`（原生回读）、`85ad67d5`（Möbius 残差）、`22009173`（单一路径）、`19cac54a`（单开关 + 去重）、`f35a9e9e`（去 `STRUCT_LAMBDA_RATIO`）、`b3178d3e`（删旧 ratio 校准）。

`question1`–`question3` 的实验与审阅按顺序归档，未完成的梯度契约要求集中转交 `question4.md`，不把部分测量修复记为完整机制验收。question1 的换码只是联合训练观测；question2 修正了占用口径与 train/eval 切换。两者原 `/tmp` 输出仍未找到，不能以最终 checkpoint 的新测量冒充旧轨迹复现。

既有日志已记录 3000 epoch 正式 Stage2 完成，实际标定为 $s=0.04444597661495209$、$\lambda=9.226019983543665\text{e-}07$；预检 λ 是其 3.81649 倍。SID SHA-256 为 `7c147fc634cc2939367d3fc83bc0a8ebfaa2d96a92cf23e060c7c88b4ac390a1`，对应分层报告已存在；这些证据不证明预检和正式运行等价，也不是本次启动或准入审计的结果。本次读取的同名 Stage3 日志最新 train 事件为 epoch 125/150，未找到最终 recall；§4.1 逐层前缀监督仍为设计项，不能宣称完整机制已有增益。

`answer3` 已持久化 JSON 与同 SHA 的脚本副本，逐层结构梯度非零及 None/zero 记录保留。但 on 公式把 `T−sg(T)` 写成外层 sg，与代码不等价；源码与独立标量代数推导显示，E16 因子的 STE 代理 Jacobian 为 encoder `3I`、C1 `−I`、C2 `0`、C3 `I`，完整 32 维范数非零不能排除半区反向或抵消。真实 RQLayer CPU 检查 120 秒超时，尚无运行验收。§3“码本 native 真梯度、encoder 直通”仍是待对齐的目标，不是当前已通过的结论；工作树数学文件仍标 DRAFT，旧运行的冻结身份仍 UNAVAILABLE，不倒填。

唯一开放追问：[`question4.md`](question4.md)——明确原生回读的目标代理 Jacobian，用解析反例和真实接线证据处理必要修复；不以调 λ 或新增损失掩盖抵消，不重训、不覆盖旧运行。原始 answer 文件不改写；完整机制与收益证据未齐，不创建新 idea。
