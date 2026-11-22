## 方案二已实现 → **encoder 步长仍然压垮 separation**,给出可测量的预算

### 已实现的改动(按你的选择)

`route_cost_gradient` 不再产生码本梯度,`run_structure_step` / trainer 显式把
`rq.vq_layers[*].embed.weight.grad = None`(**不是置零**——置零仍会让 weight decay 在 step 里作用)。
单测 23 项全绿,新增断言:

```python
assert report["codebook_frozen"] is True
assert all(model.rq.vq_layers[l].embed.weight.grad is None ...)
```

实测确认码本一步未动:

```
moved encoder.mlp.10.weight   1.004e-03
moved encoder.mlp.7.weight    1.003e-03
codebook moved? False
```

### 但 separation 仍然掉:L1 `0.5211 → 0.1626`(10 步内)

**原因不是码本,是 encoder 把 latent 整体挪出了码本所在的分辨率带。**
测得的分辨率尺度:

| 量 | 实测 |
|---|---|
| 分配给商品的码字距离 `sqrt(mean min cost)` | **0.0458** |
| encoder latent 范数 | 0.1577 |
| 码字范数 | 0.2424 |

`GRADIENT_RMS` 归一化被证明**无效**:scale = 1.0 / 0.1 / 0.01 三种,10 步后 L1 分别掉到
0.163 / 0.130 / 0.137 —— 因为 AdamW 的 `m/sqrt(v)` 让更新量级与梯度幅值无关,归一化梯度不改变步长。

### 可测量的步长预算

真正起作用的是**步长相对分辨率的比例**。按 `lr = fraction · resolution / latent_norm` 直接设定:

| fraction | 结果 |
|---:|---|
| 0.10 | 20 步内 `verify_margins` 报 **tie-decided 94.5%** |
| 0.03 / 0.01 | 待测(上一轮 0.10 就崩,说明需要更小) |

`verify_margins` 的判据是"最佳与次佳的 margin 是否可分辨";当 encoder 把商品推得彼此过近时,
`argmin` 退化成掷硬币。**这是同一条几何约束的第二个面**:separation 管码字,`verify_margins` 管商品与码字的相对关系,
两者都在说"encoder 动得太远"。

### 我现在的判断(需要你确认口径)

方案二把码本摘出去后,**encoder 这一侧仍然没有可用的稳定步长**:实测 0.10 分辨率预算会在 20 步内崩,
0.03/0.01 我会继续测。如果 0.01 也崩,说明**"用结构梯度改 encoder 参数"这条路本身**与
`assignment_floor` 的两道护栏不相容——那道护栏是按"重建损失驱动 encoder"标定的,
结构项要动同一个 encoder,就必须重新标定护栏,而那是改框架判据,不该我单方面做。

我会先把 0.03 / 0.01 两档测完,再给你一个明确结论:要么找到稳定步长并开训,要么报"方案二在现有护栏下不可行"。
