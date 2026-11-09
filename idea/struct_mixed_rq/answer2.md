# answer2 — 测量已修正；分支差异出现在 L3，几何关闭后结构项对一级码本梯度为 0

**状态：CLOSED（参与性证据已更新；下游增益仍待 Stage3）**　完成：2026-11-10　对应 `question2.md`

---

## 1. 对 answer1 六条解释的更正（原文与数值仍可追溯）

| answer1 的说法 | 更正 | 依据 |
|---|---|---|
| “96.9% 商品落在同一前缀” | 错。`same_prefix_item_share` 是**所有重复桶内商品占比**，不是最大桶占比。同一 checkpoint 上 L1 最大桶占比为 **0.5469**，重复商品占比为 **1.0000**，`1−K/N` 为 **0.9531** | `probe_measurement.occupancy`，见 §3 |
| “训练分支与导出分支 token 相同” | 错。预检只切换了 `infer_use_sk`，没有切 `model.train()/eval()`；正确测量下 **L3 有 42/64 个位置不同**，L1/L2 相同（只有末层启用 SK） | §3 分支测量 |
| “少采三元组更保守、结论更强” | 不成立。`structure_loss` 对三对残差取均值，采样影响覆盖与分配上下文，不是单调缩放 | `structure.py::structure_loss` |
| “邻居同前缀率 0.75 说明结构变好” | 分母无效。该 batch 只有 **8 条边**落在 batch 内，7 条同 L1 前缀 = 0.875；随机配对期望为 0.4370，属聚桶诊断，不是结构改善证据 | §3 边统计 |
| “预检与正式训练同一标定” | 不同。预检 `s=0.041920 / λ=3.5211e-06`；正式运行 `s=0.0444459766 / λ=9.2260e-07`，比值 **3.8165**。归因 `UNKNOWN` | 训练日志 + §3 |
| “执行证据可复算” | 原 `/tmp/q1_lite.py`、stdout/stderr 未保留，标 `UNAVAILABLE`；本次所有输入/源码 SHA 见 §5 | 本次执行身份 |

原 `answer1.md` 不改写，上述更正以本文件为准。

---

## 2. 测量器修复（TDD 红→绿，真实函数）

- **红**：`tests/test_measurement_contract.py` 首次运行 `ModuleNotFoundError: probe_measurement`。
- **绿**：新增 `probe_measurement.py`，4 条测试通过：

| 测试 | 断言 |
|---|---|
| `test_occupancy_separates_duplicate_share_from_largest_bucket` | 桶 `[56,2,2,2,1,1]` ⇒ 重复占比 `0.96875`、最大桶 `0.875`、`1−K/N` 分列 |
| `test_edge_stats_report_undefined_without_edges` | 无有效边 ⇒ `same_prefix = "UNDEFINED"`，不写 0 |
| `test_edge_stats_count_effective_denominator_and_random_expectation` | `kept=3`、同前缀 2、比例 `2/3`、随机期望 `4/12` |
| `test_branch_tokens_measure_train_mode_and_export_mode_separately` | 只有 train 分支进入 Sinkhorn（token 相同也不足以判定） |

期间修正实测缺陷：`same_prefix` 曾用 float32 掩码均值（`0.6666666865`），改为整数计数相除。机制代码（`layers.py`/`model.py`/`mixed_geometry.py`/`structure.py`）未被顺手修改。

---

## 3. 固定状态测量（交付 checkpoint，CPU，无 optimizer step）

**执行状态**：`rqvae_best.pth`（SHA `84f05d46…4672`），`load_state_dict` 无 missing/unexpected；batch 为 TRAIN 前 64 个商品（ID 0–63）；三元组请求 64、接受 **45**、接受率 **0.703**；`BEHAVIOUR_ON=True`、`GEOMETRY_ON=True`。

### 3.1 分支（train vs eval+infer_use_sk=False）

| 层 | 不同位置数 |
|---|---|
| L1 | 0 |
| L2 | 0 |
| L3 | **42 / 64** |

### 3.2 占用（train 分支；括号内为 export 分支不同处）

| 层 | 前缀数 | 最大桶占比 | 重复商品占比 | `1−K/N` | 桶（前 5） |
|---|---|---|---|---|---|
| L1 | 3 | 0.5469 | 1.0000 | 0.9531 | 35, 24, 5 |
| L2 | 33 | 0.1094 | 0.6875 | 0.4844 | 7, 5, 4, 4, 4 |
| L3 | 64 | 0.0156 | 0.0000 | 0.0000 | 1×64 |
| L3（export） | 58 | 0.0469 | 0.1562 | 0.0938 | 3, 3, 2, 2, 1 |

即：L1 只有 3 个码、最大桶 54.7%；重复占比 1.0 并不等于单桶。

### 3.3 有效边（batch 内）

| 层 | 保留边 | 同前缀边 | 比例 | 随机期望 |
|---|---|---|---|---|
| L1 | 8 | 7 | 0.875 | 0.4370 |
| L2 | 8 | 1 | 0.125 | 0.0322 |
| L3 | 8 | 0 | 0.000 | 0.0000 |

### 3.4 固定状态梯度（`s=0.041920`、`λ=3.5211e-06`）

| 块 | `L_base` 范数 | `λL_struct` 范数 | 合成范数 |
|---|---|---|---|
| encoder | 2.0175e-03 | 6.9914e-04 | 2.1379e-03 |
| codebooks | 6.0621e-04 | 3.5789e-04 | 6.5727e-04 |
| decoder | 4.3197e-04 | **0.0** | 4.3197e-04 |

encoder 上 base 与结构梯度余弦 **−0.0154**（近正交）；`L_base=2.7261e-04`、结构项 `5.7529`。decoder 结构梯度为 0 符合设计（decoder 不在结构通路）。

### 3.5 开关（同一固定状态，仅前向/反向，不 step）

| 开关 | 结果 |
|---|---|
| `behaviour_off` | 结构项 `0.0` 且 `requires_grad=False` ⇒ 梯度确实移除 |
| `geometry_off` | token 与开启时**相同**（0/0/0 变化）；结构项 `5.7387`（vs 5.7529）；**一级码本梯度范数 = 0.0** |

`geometry_off` 下结构项仍非零却对一级码本无梯度。已定位原因并用独立测试确认：

- geometry 开启时 `levels[-1]` 来自原生回读，码本依赖真实存在（三级码本均收到非零梯度：实测 3/3）；
- geometry 关闭时 `levels[-1]` 是切空间累计值，而 `VQLayer.forward` 的直通估计是 `x_q = x + (x_q - x).detach()`，**码本被该 `detach` 截断**；实测对三级码本参数的梯度全为 0。

因此这不是开关接线错误，而是"欧氏对照臂的读回本身对码本不可微"这一设计事实：该臂的结构项只能经编码器直通训练 encoder，无法训练码本。若将来要训练 geometry-off 臂，需在规格里显式定义其读回与梯度边界；本次不训练任何关闭臂，故不修改实现。

---

## 4. 预检与正式运行的对齐（差异分层，不编造归因）

| 项 | CPU 预检 | 正式运行 |
|---|---|---|
| `s` | 0.041920 | 0.04444597661495209 |
| `λ` | 3.5211e-06 | 9.226019983543665e-07 |
| 每步三元组 | 32（question1）/ 64（本问） | 256 |
| batch | 64 | 4096（4×1024） |
| 初始化 | 交付 checkpoint | KMeans + Xavier（训练时） |

`λ` 比值 **3.8165**。初始化、样本、batch、DDP 状态均不同，**归因 UNKNOWN**；不做重标定挑值。

---

## 5. 执行身份

| 项 | 值 |
|---|---|
| 脚本 | `stage2_RQ-VAE/TIGER_struct_mixed_rq/precheck_question2.py` |
| 命令 | `python3.9 precheck_question2.py`（`genrec_env_v2`） |
| checkpoint SHA-256 | `84f05d46419e6f5c8b80706bd24aafbf41a401cf8d6e5631e99305eb98c14672` |
| SID SHA-256 | `7c147fc634cc2939367d3fc83bc0a8ebfaa2d96a92cf23e060c7c88b4ac390a1` |
| `model/layers.py` | `51b287f6e280b09a5d3813d3b0050dda94b2cbec8edcda4eb1538b4a5b4f95d3` |
| `model/model.py` | `ba221245c40fd1083d2034b5835ef636149d3bdda1bbb4ab29e4b2f9f7a49c2b` |
| `mixed_geometry.py` | `7f9ee28f7c46e3697e93ae393ebedc82c0870732a12dd739194d810ae9a0cc08` |
| `structure.py` | `bbbb79caf9392829a87ff9d5494a113502445eabc8bd0831d652ee1149bfc3d6` |
| `training_hooks.py` | `a071174646a8bf6083e1e5eb17e75bd408ecbceabb286c2b26332e251d7539a6` |
| seed | 42 |

---

## 6. 四个结论分开写

1. **训练轨迹发生变化**：`answer1` 的换码计数保留为原执行者报告（`UNAVAILABLE` 复现材料），本问未重跑联合训练。
2. **结构项有梯度**：固定状态实测 encoder `6.99e-4`、codebooks `3.58e-4`，decoder 为 0。
3. **曲率计算影响当前前向/分配**：部分成立——分支差异只在 L3（42/64）；`geometry_off` 使 token 不变、但结构项对一级码本梯度归零，需定位。
4. **推荐有增益**：**UNKNOWN**。Stage3（`TIGER_struct_mixed_rq`，150 epoch，seed 42，beam 20）仍在训练；`test_R@10` 尚未产出，不作任何增益声明。

---

## 7. 关闭条件核对

| 条件 | 状态 |
|---|---|
| 原回答的解释已更正 | ✅ 第 1 节 |
| 测量回归有真实红→绿 | ✅ 第 2 节（红=ModuleNotFound，绿=4 passed） |
| 固定状态真实模型接线结果与执行材料 | ✅ 第 3、5 节（含 `geometry_off` 的零梯度发现，未当通过） |
| 预检与正式运行的标定差异分层说明 | ✅ 第 4 节，归因 `UNKNOWN` |
| 参与性与下游增益分别下结论 | ✅ 第 6 节；增益待 Stage3 |

**未被本问关闭的项**：`geometry_off` 下结构项对一级码本梯度为 0；逐层前缀监督（§4.1）仍为设计未实现。
