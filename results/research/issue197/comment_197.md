## 完成：Stage2 + Stage3 单次冻结运行，按预注册**落入关闭分支**，允许类**以零增益结案**

### Stage3 结果（`test_final.json` 全部键）

| 键 | 值 |
|---|---|
| `n_eval` | 57,439 |
| `test_recall@5` | 0.03762252128344853 |
| **`test_recall@10`** | **0.056094291335155554** |
| `test_ndcg@5` | 0.025119059325818382 |
| `test_ndcg@10` | 0.031070123887301286 |

文件：`results/stage3_T5Train/TIGER_sign_rq/logs/Amazon_2023_Instruments/Nov-12-2026_13-24-52/test_final.json`。

### 预注册裁决

- **`≤ 0.056216` ⇒ 本线关闭**：实测 **0.05609429 ≤ 0.056216**（低 **0.000122**，相对 **0.22%**）⇒ **关闭**。
- 并按本 Issue 边界条款：本臂仍 ≤ 基线 ⇒ **允许类在"结构 → 显式双曲运算 → 选码 → SID"上以零增益结案，不再开同类臂（不做消融）**。

### 为什么是负结果（与裁决并列，不改判）

- **Δ 在派发前就已退化**：`Δ>0` 仅 **28 / 13,714**（0.2042%），对 **99.9%** 商品一级曲率恒为 1 ⇒ 机制在这些行上与冻结基线逐位相同。该结论**在训练之前**记录于 `results/research/issue197/FINDINGS.md` §1–5，并按 Issue 要求写成"该臂等价于 X"。
- **一级塌缩**：L1 只用 **48/256** 码、最大码 **11.37%**；有效 L1 Hard ER 3.4689 是粗桶纯度，负命中率升至基线 **11.8 倍**（0.05804 vs 0.00491）。
- **第三码不增益**：`Conditional G3 = −0.18277`，本次会话最负。
- 即：**不是双曲运算有害，而是结构信号在这张图上几乎不产生正段**，机制基本未被激活。

### 验收对照（8 项）

1. `BEHAVIOUR_OFF` ⇒ 全体 `κ=1`，逐项断言 + off 时机制不可达。
2. `Δ_i` 独立复算（BFS + 显式循环，200 件）⇒ **0 处不符**（含 `None`）。
3. `SIGN_FLIP` ⇒ 启用集合为**精确补集**（集合级断言 + 检测器自身的负对照）。
4. 两段/缺失占比落盘：28 / 13,660 / 10,842；**`degenerate_single_regime = true`**，已如实标注。
5. κ=1 与 `poincare_contract` **maxdiff 0.0（rtol 1e-6 通过）**；κ∈{1, 8.2671, 0.001} 与边界梯度有限；**gradcheck 通过**。
6. 冒烟 **`SMOKE_PASS`**（第 4 次派发；前 3 次各暴露一个真实缺陷：`[B,B]` 36 GiB OOM、Möbius 分母下溢、冻结表 0 曲率）＋导出 `collision = 0.3080896409 < 0.40` **通过**。
7. §63 分层记录 `stage2_metrics/8e9b097b….json`；§6b 归档 **`r2:genrec/73/`**（19 文件：含 `sign_metric.py` 的源码、SID 表、checkpoint、recall 全 6 键），逐文件 SHA-256 入 `r2_ledger.json` 的 `73` 项。
8. 三次冒烟失败各留可重现用例；树内 `tests/test_sign_metric.py` **13 项全绿**。

### 附：两处继承事实

- **基线的既有红测**：`test_prefix_eligibility.py` + `test_support_regressions.py` 在 `TIGER_mixed_rq` 与本树**同为 9 failed / 16 passed**，形状相同 ⇒ 起点本身就红，非本 Issue 引入。
- **`Δ` 构建 fail-open**：训练帧缺 `seen_history` 时 `_delta = {}` 并记录 `delta_unavailable`（绝不凭空造 Δ），改后 `test_trainer_smoke.py` 6 项全过。

### 依据 commit

`99d7b7bac` · `24a4950c7` · `3ee1c61ab` · `4b611d6e7` · `091a38127` · `9fce6bd00` · `80715a0e0` · `d3bbab255` · `f799f883c` · `86e71088a` · `d343b98ff` · `612b384db` · `933943acd` · `a0cbf8891`（结果、归档、关闭依据）。
