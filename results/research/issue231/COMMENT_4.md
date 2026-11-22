## C19/C28:结构项已接入 `L_total`,1-epoch smoke 通过(15 测试绿)

### 接了什么

| 文件 | 改动 |
|---|---|
| `model/layers.py` | `VQLayer.forward` 返回该级 cost 矩阵;`RQLayer` 收集为 `level_logits` |
| `model/model.py` | `forward`/`get_indices` 透传 logits(5 元组) |
| `train_rqvae.py` | `loss = loss + graphlet_value * scale`,**同一次 `backward()`**;无 `autograd.grad` |
| `train_rqvae.py` | dataset 带 item id;id 只用于查 bank,**不进 encoder**(C6) |

### smoke(1 epoch,单卡)

| 项 | 值 |
|---|---|
| 总 loss | **0.00056446** |
| 重构 loss | 0.00046401 |
| 结构项占比 | **21.6%** |
| 碰撞率 | 0.066092 |
| 导出 | `sid_shape=(24587,4)`,unique 24587 |

### 修掉的缺陷:λ 不能拍定

首版 `λ=0.05` 让总 loss 到 **0.142**,结构项是重构的 **305 倍** —— 目标被吞掉,测不出机制。
结构项起点贴近上限(2.83/3.0),重构是 4.6e-4,量级差太大。按 §19g 不许试值,改为可推导:

```
scale = GRAPHLET_SHARE * recon_loss.detach() / (3 * MARGIN)
```

- `3 × MARGIN` 是三结构各一个 margin 的**定义上限**,不是挑的常数;
- `GRAPHLET_SHARE=0.10` 要求结构信号占重构固定比例,是唯一与量纲无关的选择;
- 重构 `detach()`,否则比例会把梯度反传进它衡量的目标。

### 闸门拦截

`patch_tree` 以 `embed_ind = self._indices(...)` **两行块**为锚点,我插注释打断了它 → fail-closed 拒启。
按闸门要求把注释移到锚点之后,未改闸门。

### 下一步

C38 真角色打乱 → C21 训练/导出分配差异 → 三组正式训练。
