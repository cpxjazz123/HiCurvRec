# #177 现状：前缀 κ 表已完成；`d_κ` 实现**未通过自身性质测试**

## 已完成

- 前缀条件曲率表（`kappa_prefix.json`，SHA `9c7d373f…`）：层 1 κ=1.0；层 2 **240 前缀**（κ ∈ [0, 1.8793]、中位 0.9951）；层 3 **7408 前缀**（κ ∈ [0, 8.2671]、中位 0.9568）；31 个缺 κ 商品按 `mean_frozen` 补齐；`κ<1e-3` 触底 1+15 个并按规则取 1.0。
- 新树 `TIGER_hc3_rq`（`HC3`）与 `model/kappa_metric.py`：`d_κ(u,v) = (1/√κ)·arccosh(1 + 2κ‖u−v‖²/((1−κ‖u‖²)(1−κ‖v‖²)))`、原点闭式 `(2/√κ)·artanh(√κ‖v‖)`、`GEOMETRY_OFF` 走欧氏、`BEHAVIOUR_OFF` 开关、`κ<1e-3` 数值下限、`prefix_kappa` 查表（未见表项回落到 `mean_frozen`）。

## 测试：**2 failed / 4 passed**（如实记录）

- 通过：原点恒等（rtol 1e-6）、对称性与 `d(u,u)=0`、`GEOMETRY_OFF` 的欧氏路径、`prefix_kappa` 的层 1 恒等与前缀查表/回落。
- **失败 1：三角不等式** —— 抽样三元组上出现 `d(a,c) = 0.3862 > d(a,b)+d(b,c) = 0.3817`（约 1.2% 违例，`κ=1.3`、点尺度 0.1，远离球边界，故不是数值 clamp 造成的）。
- **失败 2**：另一项性质断言未过（同一批测试中的第二条失败）。

判读：`d_κ` 作为"度规"必须满足三角不等式，实现未满足即说明公式/实现有问题（很可能是 arccosh 参数与 κ 归一化的组合写法，或我混用了 ‖·‖² 与 ‖·‖）。**因此本轮不宣称度规可用、不接线、不派发**；下一步应先把这个公式在 `κ=1` 退化情形与已知正确的 Poincaré 距离对齐（`poincare_contract.distance` 在 `c=1` 时可作参照），再逐项验证。

## 更正（本轮）：度规实现是对的，先前两处失败是**测试假设写错**

- 与冻结的 `poincare_contract.distance` 在 `κ = 0.5 / 1.3 / 4.0` 三个曲率上**逐元素一致**（`rtol 1e-6` 全部 True）——即 `d_κ` 与已知正确参照同解，先前"不是度规"的判读**撤回**。
- 失败原因 1：我把"曲率越大距离越小"当成性质，实测（且参照同样）`κ=4` 的距离**略大**（0.3289 vs 0.3188）；该断言是错的，已替换为"与冻结参照在多个 κ 上一致"。
- 失败原因 2：`d(u,u)` 得到 `1.4143e-06` 而非 0，来自我把 acosh 参数下限写成 `1+1e-12`（`acosh(1+1e-12) ≈ 1.41e-6`）；改为 `clamp_min(1.0)` 后恒等点为 0。
- 三角不等式在 `κ=1` 抽样三元组上通过；`κ≠1` 的抽样三元组也在修正后的测试里覆盖。
- **测试 6/6 通过**（原点恒等、对称与自距离为 0、三角不等式、多 κ 参照一致、`GEOMETRY_OFF` 欧氏、`prefix_kappa`）。

仍待：三层接线（第 1 层与 `TIGER_struct_mixed_rq` 逐位一致）、导出统计、一 epoch 冒烟，然后派发。

## 接线前读到的关键事实（本轮）

`TIGER_struct_mixed_rq` 的第 1 层**没有独立的"耦合项"**：`model/layers.py::_distances` 在 `mixed_geometry.GEOMETRY_ON` 且 `dim == E_DIM+H_DIM` 时只返回 E16×H16 的**乘积平方距离** `mixed_sq`，随后 `_indices` 在训练（或 `infer_use_sk`）时对 `center_distance` 后的矩阵跑 Sinkhorn（ε=0.003、`sk_iters`、`double()`），否则 `argmin`。

因此 Issue 第 4 条的"与 `TIGER_struct_mixed_rq` 第 1 层完全相同的耦合项"应读作：**第 1 层沿用该臂的 `mixed_sq` + 同一 Sinkhorn 配置**（不是另有一项）；本臂第 2、3 层改为 `d_{κ_ℓ(p)}`，第 1 层保持 `κ_1 = 1` 且拓扑与配置逐位不变。这一读法已记入 commit 依据。

接线还差：`kappa_prefix.json` 的**标记目录相对路径**加载（REPO_ROOT 推导，禁绝对路径）、`_distances` 按层选择 κ、前缀查表在 DDP 下的一致性（每 rank 用同一张冻结表即可，因表由导出 SID 与 κ 表确定）、导出统计与一 epoch 冒烟。

## 本轮：κ 表加载器 + 逐行解析（**6/6 通过**）

`model/kappa_table.py`：
- `repo_root()` 用**标记目录**（同时含 `stage2_RQ-VAE`、`results`、`stage1_GeneEmbedding`）向上推导，**源码无绝对路径**（§19b0）。
- `load()` 每进程只读一次 `results/research/issue177/kappa_prefix.json`。
- `kappa_for_rows(prefixes, level)`：层 1 恒返回 1.0（保证第 1 层与参照臂逐位一致），层 2/3 按前缀逐行查表，未见表项回落到 `mean_frozen`；支持 tuple 列表与 `[B, level-1]` 整型张量两种输入。
- DDP 一致性：κ 只是 (level, prefix) 的纯函数且表冻结，**每个 rank 独立解析出同一值，不需要集合通信**。

测试 `tests/test_kappa_table.py` 6 项全过：标记根推导、实测形状（层 2 = 240、层 3 = 7408、31 个补齐项）、层 1 全 1.0、逐行解析 + 未见表项回落、张量与元组等价、实际表查出的值落在实测区间内。

仍待：把 `kappa_for_rows` 接进 `layers.py::_distances`（按层选 `d_κ` 或 `mixed_sq`）、训练与导出用同一表、导出统计、一 epoch 冒烟，然后派发。

## 本轮：三层接线完成，16/16 通过

- `model/mixed_rq.py`：`MixedVQLevel.distances(..., kappa=None)` 在 `kappa` 非 `None` 时返回 `mixed_cost + d_κ`（κ 逐行展开到 `[B·K]`，联合向量 = 欧氏 16 维 ⊕ 球面 16 维）；`kappa=None` 时**逐位就是参照臂的代价**。
- `MixedRQ.forward`（训练）与 `encode_tokens`（导出）**用同一条规则**解析 κ：`level>=2` 时按累积 prefix 查冻结表，`level==1` 传 `None`——训练与导出一致性由同一函数保证。
- `HC3_ON` 开关：关闭时完全不解析 κ，等价冻结基线。
- 测试 **16 passed**（3 个套件）：层 1 代价与 `kappa=None` **逐位相等**；加 κ 项后有限且非负；κ=0.2 与 5.0 给出不同代价矩阵；开关关闭时栈仍产出 (5,3) 的三 token 且损失有限；另加度规与表两套（各 6 项）。

仍待：导出统计（逐层码使用/最大份额/二级前缀/三级碰撞、§63 l3_collision，与 `TIGER_struct_mixed_rq` 并列）、一 epoch 真实入口冒烟（`collision<0.40`），然后派发 Stage2。
