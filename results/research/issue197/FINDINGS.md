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

## 8. 冒烟通过（第四次派发，前三次各暴露一个真实缺陷）

`SMOKE_PASS`，训练已起（6 进程，epoch 145 时 loss 1.5234e-04）。三次失败与修复，全部是工程/数值缺陷，不是机制判断：

| 次序 | 失败 | 根因 | 修复 |
|---|---|---|---|
| 1 | `CUDA OOM，36 GiB` | commitment/codebook 两项把"每行选中的码"喂给按码广播的 `curved_squared` ⇒ `[B,B,D]` | 新增 `curved_squared_paired`（逐行、返回向量），并留回归用例 |
| 2 | `loss became non-finite at epoch 1` | 逐行 Möbius 分母在球边界趋于 0，原先 clamp `1e-30` ⇒ 1e30 级商值 | 分母下限 `1e-3`（内部点永不触发，κ=1 对账仍 rtol 1e-6） |
| 3 | `curved cost not finite (kappa range 0.0..8.267)` | 冻结 AF3 表含 0 曲率条目，`2/√0 = inf` | `KAPPA_FLOOR = 1e-3`，复用 #180 `r48k_geometry` 对同一张表的既有下限 |

三次都由冒烟本身发现并阻断（这正是冒烟的作用），每次修复都留下能重现该形态的用例；树内 `tests/test_sign_metric.py` 现为 **13 项全过**。

## 9. Stage2 完成：导出统计与 §89 门

3000 epoch 完成（`wall_time ≈ 629.7 s`），`item_sids.json`（24,587 商品，行宽 4），`post_training_export.json`：

| 指标 | 本臂 SIGN | 参照 |
|---|---|---|
| **§89 门 `collision`（门算式）** | **0.3080896409** | 门限 0.40 ⇒ **通过** |
| 不同三级组合 | 17,012 | SDH 19,377 · #192 12,186 |
| 二级前缀数 | 3,081 | 基线 15,860 · SDH 6,287 |
| **L1 使用码数** | **48 / 256** | SDH 246 · #192 142 |
| **L1 最大码占比** | **11.37%** | SDH 3.96% · #192 9.10% |
| L3 最大码占比 | 0.146% | — |
| L3 熵 / 最大值 | 97.55% | — |

§63 分层记录（`stage2_metrics/8e9b097b….json`）：

| 指标 | 本臂 | 基线 |
|---|---|---|
| 有效 L1 Hard ER | **3.4689** | 16.1417 |
| L1 正/负命中率（仅诊断） | 0.20132 / **0.05804** | 0.07931 / 0.00491 |
| L2 前缀数 | 3,081 | 15,860 |
| 不同三级组合 | 17,012 | — |
| §63 `l3_collision`（另一口径） | 0.45272 | — |
| L3 Refinement Gain (bits) | 3.62676 | — |
| Conditional G3 | **−0.18277** | — |

读法（与裁决并列、不替代）：碰撞门通过，但**一级码本严重塌缩**——256 个一级码只用了 **48** 个、最大码占 **11.37%** 商品；有效 L1 Hard ER 低到 3.47 正是塌缩的产物（粗桶更纯），同时**负样本前缀命中率升到基线的 11.8 倍**（0.05804 vs 0.00491），即一级已基本失去区分力。`Conditional G3 = −0.1828` 是本次会话所有臂中最负的一档，第三码在给定前两码后不仅不增益、且倒退更多。这些是前导诊断（§19/§63 不作闸门），但它们与"该臂对 99.9% 商品就是基线几何"一致：塔基已经被压扁，而结构信号只碰到 28 个商品。

## 10. Stage3 已派发

`stage3_start` 返回 `started: true`，pid **4123013**，variant `TIGER_sign_rq`（与 Stage2 臂目录同名，§64b 满足）；冻结协议核验通过（150 epoch、`NO_EVAL=True`、`SKIP_TEST=False`、seed 42、beam 20、无 early stop），分层记录 `passed: true` 复用 `8e9b097b…json`。Stage2 的 `train_end` 自报 `final_collision = 0.30808964086712487`、`unique_after_extend = 24587`，与门算式一致。
