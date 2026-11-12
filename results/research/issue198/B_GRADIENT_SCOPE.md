# #198-B 梯度与曲率作用范围（`audit_b.json`）

## 1. 挂点与作用范围（源码行号，逐条引用）

| 位置 | 行 | 内容 |
|---|---|---|
| `model/model.py` | 93–94 | `kappa = sign_metric.regime_kappa(item_ids.cpu(), self.delta)`；`self.rq(encoded, kappa=kappa)` |
| `model/model.py` | 108 / 116 | `get_indices` 与 `init_codebook` 同样按 ids 取 κ |
| `model/mixed_rq.py` | **377 / 397** | `layer(..., kappa=(kappa if level == 1 else None))` —— **唯一的分层路由点**，二三级显式传 `None` |
| `model/mixed_rq.py` | 181 | `mixed_cost_with_kappa(...)` 用于一级 `distances` |
| `model/mixed_rq.py` | 231 | commitment / codebook 两项用 `curved_squared_paired` |
| `model/sign_metric.py` | 136 / 146 / 150 | `regime_kappa` 定义、`Δ>0` 判定、`clamp_min(KAPPA_FLOOR)` |

**结论：曲率只进入一级**（377/397 两处路由把二三级置 `None`）。`#193` 发现的"通用 `VQLayer` 挂钩"在**本树不存在**：`sign_metric` 是被 `mixed_rq` 的 `mixed_cost_with_kappa` 显式调用的，不是模块级全局状态。

## 2. 分块梯度范数（真实 checkpoint、真实 batch 1024、CPU）

| 块 | 门开（两段式） | 门关（κ≡1） | 相对差 |
|---|---|---|---|
| loss | 1.4363329682964832e-04 | 1.4363348600454628e-04 | 1.3e-06 |
| encoder（8 张量） | 5.521917038130876e-05 | 5.5220680378436584e-05 | **2.7e-05** |
| decoder（8 张量） | 5.6709956203489315e-05 | 5.6709956203489315e-05 | 0 |
| codebook（3 张量） | 5.143974037466332e-05 | 5.144032426964409e-05 | **1.1e-05** |

同一 batch 下，门开/门关的梯度范数只差 **1e-5 量级**——与"1024 行里平均 1.1 行受影响"完全一致。

## 3. 28 件商品的 κ 与 batch 命中

- 启用件数 **28**；`kappa_rows_uniform_one_share = 0.99886`（99.886% 的行 κ≡1）。
- 28 件的 κ：min **0.0**、q25 0.0294、median **0.0959**、q75 0.3883、max 8.2671、mean 0.5351。
  ⇒ 多数启用件的 κ **远小于 1**（曲率弱于基线！），只有个别接近 8.27；且最小值为 0（使用时被 `KAPPA_FLOOR=1e-3` 兜住）。
- 每 batch（1024）命中数：样本 `[2,0,1,2,0,0,2,3,1,1,0,1,0,1,1,1,0,2,2,2]`，均值 **1.1**、最大 3、**20 个 batch 中 6 个命中为 0**；理论期望 1.168。

## 判定

**（ii）曲率作用范围泄漏 ⇒ `EXCLUDED`。** 一级路由是显式的（`mixed_rq.py:377,397`），二三级恒 `None`；同 batch 的分块梯度范数在门开/门关之间只差 1e-5 量级。

**产物 `kappa_rows_uniform_one_share = 0.99886` 与"28/24,587 = 0.114%"互证**，说明机制的作用面确实只有这 28 件。
