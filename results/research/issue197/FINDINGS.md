# Issue #197 — 派发前的实测：**Δ 退化为单一 regime（正段 0.204%）**

`results/research/issue197/sign_shares.json`（TRAIN 图，`results/stage0_build_parquet/train.parquet`）：

| 量 | 值 |
|---|---|
| 训练边 / 三角形闭合边 | 323,633 / **193,495** |
| 三角形闭合子图的连通分量数 | **7** |
| 落在分量内的商品 | 18,643 / 24,587 |
| `Δ` 可定义 / 不可定义 | 13,714 / 10,842（缺失占已触及 44.15%） |
| **`Δ > 0`（启用 κ_i 的商品）** | **28** |
| `Δ < 0` | 13,660 |
| **正段占可定义** | **0.2042%** |
| **`degenerate_single_regime`** | **true** |
| `Δ` 分位（q05 / q50 / q95 / mean） | −0.2305 / −0.1130 / −0.0144 / −0.1167 |
| 独立复算（BFS 分量 + 显式循环平均，200 件） | **0 处不符** |

## 按本 Issue 预注册的读法

Issue 明确：若 `Δ` 在训练集上几乎全正或全负（退化为单一 regime），**冒烟必须给出占比，并在报告中如实写"该臂等价于 X"**。

这里的 X 是：**"冻结基线 + 在 28 个商品（0.114%）上把一级双曲曲率从 1 换成其冻结 AF3 κ"**。其余 24,559 个商品一级曲率恒为 1，机制在这些行上与基线逐位相同。

## 成因（记录，不改机制）

- 三角形闭合子图几乎全连通（**7 个分量**把 18,643 个商品连在一起），因此"组"几乎是整张图，跨组邻居极少；
- 组内平均权恒 ≥ 跨组平均权 ⇒ `Δ` 系统性为负；只有处在分量边界、且跨组边权重明显更重的 28 个商品为正。

## 独立复算（验收 2）

第二个实现故意写得笨拙：从 parquet 重建边、用 BFS 求分量（而不是并查集）、用显式循环求组内/组外均值。抽 200 件，与并查集实现**逐件比对，0 处不符**（含 `None` 的一致性）。

## 6. 训练器的 Δ 构建必须对合成帧 fail-open

首次跑树内全量测试时 `tests/test_trainer_smoke.py` 有 4 项 `KeyError: 'seen_history'`——冒烟夹具构造的 `train_frame` 没有该列，而我在训练器里直接读了它。已改为：缺列 ⇒ `_delta = {}`、`_shares = {"delta_unavailable": ...}`，臂在基线几何上运行并**记录原因**，不凭空造 Δ。改后 `test_trainer_smoke.py` **6 项全过**。

## 7. 继承基线的既有红测（不是本 Issue 引入）

`tests/test_prefix_eligibility.py` 与 `tests/test_support_regressions.py` 在两棵树里**逐项相同地失败**：

| 树 | 结果 |
|---|---|
| `TIGER_mixed_rq`（冻结起点，只读） | **9 failed / 16 passed** |
| `TIGER_sign_rq`（本树） | **9 failed / 16 passed** |

失败形态相同（`test_stale_explicit_local_entry_stops_before_any_selection` 的 8 个参数化用例未抛 `RuntimeError`、`test_proportional_quotas_are_enforced…` 断言为假）。按规则不修改、不重新钉住基线既有测试；此处只记录"起点本身就红"，以免被当作本 Issue 的回归。
