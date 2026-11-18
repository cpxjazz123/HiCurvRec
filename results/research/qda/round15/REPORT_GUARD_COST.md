# 保护层的训练开销(实测)

3000 epoch 的臂最关心的一件事:这层保护要花多少时间。

## 方法

冻结基线的同一套配方、同一 seed,60 epoch;三条臂:

| 臂 | 内容 |
|---|---|
| `PLAIN_60` | 原样基线 |
| `FLOOR_60` | 只装分配下限(去掉每步的两个探测器) |
| `GUARDED_60` | 完整保护(下限 + `verify_codebook` + `verify_margins` + `verify_assignment`) |

所有输出路径重定基到本轮目录,不触碰任何已发表结果。

## 实测(steady-state 每 epoch 秒)

```
PLAIN_60    0.1814 s/epoch
FLOOR_60    0.2112 s/epoch     +16.4 %
GUARDED_60  0.2141 s/epoch     +18.0 %   (探测器在下限之上只加 +1.4 %)
```

结论:**开销几乎全部来自分配修复本身**(+16.4 %),每步的两个探测器只占 +1.4 %。也就是说,这 16 % 买的是"每个码字每步都有商品"这个保证;探测器几乎是免费的。

按冻结配方(混合臂约 0.43 s/epoch、3000 epoch ≈ 22 分钟)估算,完整保护约多花 4 分钟。

## 顺带做的两处优化(都保留)

1. **健康路径不付修复成本**:`constrained_assignment` 先做一次 `bincount`,满足下限就直接返回,不再把 4096 个索引搬到 host。
2. **修复循环向量化 + 用基线自身的 float32**:原来是 `.double()` 加逐码字的 Python 列表操作;现在每轮用张量切片取"空缺码字 × 可让出商品"的代价,`.tolist()` 只发生在 256 个码字上。语义由 29 项测试覆盖。

## 复现

```bash
<genrec_env_v2>/python3.9 results/research/qda/round15/bench_overhead.py
```

注意:两次运行之间有 ~5 % 的抖动(第一次 30 epoch 的测量给出 +28 %,60 epoch 复测为 +18 %),因此报告用的是 60 epoch 的 steady-state 均值。
