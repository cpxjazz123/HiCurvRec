## §0/§1 冻结清单 83 条 + 崩塌已定位到 **L2**

正文 sha256 `d9f3a653d55c04648bccc1f979bcc75db3f44bcd8d8a17966911952e352a9d7e`,3,353 字符。
清单 `results/research/issue233/RUN_LOG.md`,**C1–C83**。

### §1 归档(C9/C10)

`results/research/issue233/run232_archive/`:

| 文件 | 字节 | 说明 |
|---|---:|---|
| `source.tar.gz` | 70,465 | `train_rqvae.py` + `train_config.py` + `config/` + `model/` + `losses/` + `tests/` |
| `training_metrics.jsonl` | 708,975 | 3000 epoch 逐轮 |
| `item_sids.json` | 1,263,713 | 最终导出 |

归档的 `train_rqvae.py` **不含 `_structure_step`**(C10),含 `loss = loss + structure_alpha * structure_value`
与 `clip_grad_norm_`(C11/C25),**无 `normalise_gradients`**(C12)。

### §1 崩塌定位(C15–C17):在 **L2**

| 层级 | 本臂 | TIGER | 比值 |
|---|---:|---:|---:|
| L1 distinct | 256 | 249 | 1.028 |
| **L2 distinct** | **3,091** | **7,374** | **0.419** |
| **L2 最大桶** | **4,862** | **85** | **57.2×** |
| **L2 条件熵** | **3.524 bits** | **4.979 bits** | **0.708** |
| L3 distinct | 7,859 | 20,425 | 0.385 |
| L3 条件熵 | 2.923 bits | 2.203 bits | **1.326** |

**三条可测量的事实**:

1. **L1 熵反而略高**(8.000 vs 7.960),码本 256 个全用上——不是"码本没用完"
2. **L2 把 4,862 个商品堆进同一个前缀**,而 TIGER 最大桶只有 85
3. **L3 的条件熵高于基线**(1.326×),说明 L3 仍在区分,是 **L2 先塌**

结论:**崩塌源于 L1 的极度不均把压力全部推给 L2**。L1 最大桶 10,700 个商品(占 43.5%)共享一级码,
L2 的 256 个码无法再区分它们,只能继续堆叠。这正是 issue §2 要求"L3 优先、不拆散已正确 L1/L2"的
数据依据——但更是"L1 就必须有容量约束"的依据。

### 预注册常量(C30/C60)

| 常量 | 值 | 依据 |
|---|---|---|
| `RAMP_START` | 1500 | issue §2"前1500 epoch纯原始 RQ-VAE" |
| `GRAPHLET_GRAD_BUDGET` | 0.10 | issue §2"不超过原 RQ-VAE encoder 梯度的10%" |
| `COLLISION_GRAD_BUDGET` | 0.10 | issue §2"独立有界梯度预算" |
| `TAU` | 0.001 | #232 实测梯度存活尺度 |

### 下一步

按 §2 实现 `L_total = L_RQVAE + λ(t)·L_Graphlet + μ(t)·L_collision`,起重放在
**L1 的容量约束**与 **L3 的重复桶惩罚**,全程梯度预算而非固定权重。
