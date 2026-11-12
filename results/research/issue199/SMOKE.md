# #199 冒烟通过（第一次派发）

`SMOKE_PASS`，训练已起（6 进程）。训练器在开训前记录的 `[l2k]` 元数据与机制定义一致：

| 键 | 值 |
|---|---|
| `items` | 24,556 |
| **`level1_uses_kappa` / `level2_uses_kappa` / `level3_uses_kappa`** | **False / True / False** |
| `switches` | `behaviour_off=False`、`level1_on=False` |
| `rows_at_baseline_curvature` | **0** |
| `rows_with_kappa_below_one` / `above_one` | 13,473 / 11,083 |
| `kappa_min` / `median` / `max` | 0.001（下限）/ 0.9147 / 8.2671 |
| `kappa_table_sha256` | `270b34f238e164950cbc3bb26acf74adc845b1a03b2ddfc62c1b6dcb588d73f2` |

两点值得记录：

1. **作用面是全体商品**（`rows_at_baseline_curvature = 0`，没有任何一行的 κ 恰好等于 1），与 SIGN 的 28 件形成对照——本臂的几何作用于二级的每一行。
2. 曲率分布在 1 两侧大致对半（13,473 低于 1、11,083 高于 1，中位 0.9147），与冻结表本身一致，未见退化。
