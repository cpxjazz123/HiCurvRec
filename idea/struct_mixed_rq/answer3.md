# answer3 — 契约与梯度通路：off 的零结构梯度符合基线 STE，开关因此不能隔离曲率

**状态：CLOSED（契约与逐层证据齐备；下游收益仍为 UNKNOWN）**　完成：2026-11-10　对应 `question3.md`

---

## 1. 真实契约与运行身份

### 1.1 身份（能核对的部分与缺口）

| 项 | 值 |
|---|---|
| checkpoint | `results/stage2_RQ-VAE/TIGER_struct_mixed_rq/rqvae_best.pth`，SHA-256 `84f05d46419e6f5c8b80706bd24aafbf41a401cf8d6e5631e99305eb98c14672` |
| SID | `item_sids.json`，SHA-256 `7c147fc634cc2939367d3fc83bc0a8ebfaa2d96a92cf23e060c7c88b4ac390a1`（由所读文件计算） |
| `run_id` | **UNAVAILABLE**：训练日志里没有 run_id，MCP 状态文件只记录 `launched_at` 与 pid |
| `math_spec.json` | SHA-256 `fc2fb26cff5577ed1580ac81ecb78447f4b332036197d5648e895877bd8faebe`；文件自称 **DRAFT / not frozen**，含旧 `ratio=0.1` 与 cumulative-codeword-sum 描述 |
| 结论 | 无法把这份 DRAFT 倒填成该次运行的冻结规格；既有运行是否持有合法冻结快照记为 **UNAVAILABLE**，不重建、不改写 |

### 1.2 on/off 两侧方程（按真实 `RQLayer`/`VQLayer` 写出）

记 `sg` 为 stop-gradient，`c_l` 为第 `l` 层选中的码字（真实 gather，不带梯度），`r_l` 为第 `l` 层输入残差，`q_l` 为该层量化输出：

**off（基线 STE）**

$$q_l = r_l + \mathrm{sg}(c_l - r_l),\qquad r_{l+1} = r_l - q_l,\qquad q_{\text{off}} = \sum_l q_l$$

- 梯度：$\partial q_{\text{off}}/\partial r_l = 1$（编码器经残差链得到直通梯度）；$\partial q_{\text{off}}/\partial c_l = 0$。
- 因此从 `q_off`（`levels[-1]`）出发的结构项**对码本没有可微路径**；这不是实现缺陷，是基线 STE 的定义。

**on（原生回读）**

$$q_{\text{on}} = \underbrace{\mathrm{Read}\big(c_1 \oplus (c_2 \oplus c_3)\big)}_{\text{原生码字回读}} + \mathrm{sg}\Big(\sum_l \mathrm{sg}(c_l - r_l)\Big)$$

- 前项对码字可微，且**前层码字通过 Möbius 合成间接影响后续位置**（不是只有最后一个码字有梯度）；
- 后项只提供编码器直通梯度。

**离散选择边界**：`argmin`/Sinkhorn 选码本身不可微；`quant_loss` 仍经 `F.embedding` 的原始 gather 值训练码本，与"经回读到码本的梯度"是两条不同通路。

### 1.3 判定（按 `question3` §3 的两种判断）

- `geometry_off` 的零结构梯度 **符合基线 STE 定义**，不是 bug；
- 因此该开关**不能隔离曲率的贡献**：它同时把回读的梯度策略换掉了，不是"只换几何、保持同一梯度策略"的对照；
- 若将来需要一个只能归因于几何的对照，须另写独立定义并做廉价前后向比较，本次不实现、不替换开关。

---

## 2. 检测器修复（TDD 红→绿，真实接口）

- **红**：`tests/test_block_measurement_contract.py` 首次运行 `ModuleNotFoundError: probe_blocks`。
- **绿**：新增 `probe_blocks.py`，4 条测试通过：

| 测试 | 断言 |
|---|---|
| `test_per_block_cosine_separates_same_and_opposite_blocks` | 两块分别同向/反向：块余弦 `+1` 与 `−1`，而全参数余弦会在 0 附近——旧实现"全参数角标成 encoder"的误标可被抓出 |
| `test_none_and_zero_are_distinguished` | `None` 张量数与"已连接但为零"张量数分别计数 |
| `test_undefined_cosine_when_a_block_has_no_norm` | 零范数块的余弦记 `UNDEFINED`，不写 0 |
| `test_the_branch_check_counts_the_real_sinkhorn` | 真实 `VQLayer`：train 模式 1 次 Sinkhorn，`eval + infer_use_sk=False` 0 次，且模式被恢复 |

另：`same_prefix` 的 float32 掩码均值问题在 question2 已改为整数计数相除。

---

## 3. 固定状态逐层真实证据（checkpoint、CPU、无 optimizer step）

身份：`s=0.04444597661495209`、`λ=9.226019983543665e-07`（正式运行标定记录），batch 为 TRAIN 前 64 个商品（ID 0–63），三元组请求 64 / 接受 **45**（0.703），`load_state_dict` 无 missing/unexpected。

### 3.1 分支（同几何）

| 层 | train vs export 不同位置 |
|---|---|
| L1 | 0 |
| L2 | 0 |
| L3 | **42 / 64** |

### 3.2 损失

| 量 | 值 |
|---|---|
| recon | 1.0600e-04 |
| quant | 1.6700e-04 |
| λ·structure | 4e-06（结构项原值 4.2938） |
| total | 2.7700e-04 |

### 3.3 逐块梯度（范数；none = 未连接张量数，zero = 已连接但为零）

| 块 | recon | quant | structure | total |
|---|---|---|---|---|
| encoder | 2.2592e-03 | 3.9775e-04 | 1.4156e-04 | 2.0231e-03 |
| C1 | 4.1725e-04 | 4.7959e-04 | 4.3310e-05 | 3.8621e-04 |
| C2 | 1.4600e-06 | 2.2375e-04 | 7.9000e-07 | 2.2375e-04 |
| C3 | 3.1431e-04 | 1.4470e-04 | 5.7250e-05 | 4.0107e-04 |
| decoder | 4.3197e-04 | 0（none=8） | 0（none=8） | 4.3197e-04 |

全部有限；结构项对 encoder、C1、C2、C3 **逐层均非零**（C2 仅 `7.9e-07`，量级最小），decoder 为未连接（none=8），与设计一致。

### 3.4 各块相对总梯度的余弦

| 块 | recon | quant | structure |
|---|---|---|---|
| encoder | 0.9865 | −0.5433 | 0.0742 |
| C1 | 0.2946 | 0.5726 | −0.2612 |
| C2 | 0.0012 | 1.0000 | 0.0053 |
| C3 | 0.9341 | 0.6868 | 0.1413 |
| decoder | 1.0001 | UNDEFINED | UNDEFINED |

（decoder 的 `1.0001` 是浮点舍入；C2 的结构余弦 0.0053 与极小范数一致。 encoder 结构余弦 **0.0742**，不是 question2 里的 −0.0154——后者是全参数合并值，已更正。）

### 3.5 同一分配模式下的 geometry on/off

| 项 | 结果 |
|---|---|
| 硬 token 差 | 0 / 0 / 0 |
| 逐层累计回读最大绝对差 | 0.0 / 1.93e-04 / 1.349e-02 |
| 结构项原值 | on 4.2938 → off 4.2829 |

即：当前样本上连续读回确实变化（L2/L3 有差），但**落在相同分配区域**，硬 token 不变。不据此宣称曲率无用，也不加大 λ 直到换码。

---

## 4. 证据落位（仓库内，非 `/tmp`）

| 内容 | 路径 |
|---|---|
| 完整诊断 JSON | `results/research/struct_mixed_rq/q3/diagnostics.json` |
| 诊断脚本副本（与实际执行同内容） | `results/research/struct_mixed_rq/q3/precheck_question3.py` |
| 脚本原件 | `stage2_RQ-VAE/TIGER_struct_mixed_rq/precheck_question3.py` |
| 检测器 | `probe_blocks.py`、`probe_measurement.py` |
| 测试 | `tests/test_block_measurement_contract.py`、`tests/test_measurement_contract.py` |

脚本/检测器/模型源码/checkpoint/SID/输入的 SHA-256 均写入 `diagnostics.json.identity`；`run_id` 为 `UNAVAILABLE`。question1/question2 的 `/tmp` 输出仍为 `UNAVAILABLE`，本问不声称复现它们。

---

## 5. 下游收益

Stage3 已完成（150/150，seed 42，beam 20，`n_eval=57439`），最终结果：

| 指标 | 值 |
|---|---|
| `test_recall@5` | 0.038301502463482996 |
| **`test_recall@10`** | **0.05790490781524748** |
| `test_ndcg@5` | 0.025511840529033068 |
| `test_ndcg@10` | 0.03183500936192825 |
| `test_final.json` | `results/stage3_T5Train/TIGER_struct_mixed_rq/logs/Amazon_2023_Instruments/Nov-10-2026_02-00-48/test_final.json` |

判定：`test_R@10 = 0.05790 < 0.065`，**未达到成功条件，属负结果**。与两条历史基线的并列比较需同名目录配对与同协议证据，本次不做“增益”声明；单次 both-on 结果也不能证明曲率的因果贡献。按 §4/§6 的边界，此负结果如实保留，不重训、不调参、不换种子。

---

## 6. 关闭条件核对

| 条件 | 状态 |
|---|---|
| 契约与两种梯度路径解释清楚 | ✅ §1（on/off 方程与 STE 边界；off 零梯度符合基线定义，不能隔离几何） |
| 测量器正确（红→绿、真实接口、None≠0） | ✅ §2 |
| 固定状态逐层真实证据与身份材料 | ✅ §3、§4（run_id `UNAVAILABLE` 明确保留） |
| 失败/未运行明确保留 | ✅ §5（Stage3 未完成；`math_spec` 仍 DRAFT） |

**未关闭的机制项**：逐层前缀监督（主说明 §4.1）仍为设计未实现；完整机制与最终收益证据未齐前不创建新 idea。
