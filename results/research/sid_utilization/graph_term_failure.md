# 一级行为约束(L_graph^(1))试跑记录:码本坍缩

## 配置

- 树:`stage2_RQ-VAE/TIGER_sinkhorn_all_levels_rq`(三层 Sinkhorn)+ 新增
  `structure_loss.py`;
- `L = L_RQVAE + λ·L_graph^(1)`,`L_graph^(1) = mean‖z_i−z_j‖² / mean‖z‖²` 于行为图边上;
- λ 在第一步标定:`λ = 0.1·r₀ / (0.9·g₀)`,实测 **λ = 1.182e-4**(r₀≈4.5e-4,g₀≈0.4275);
- 边来自 `results/research/behavior_graph/graph.npz`,sha256 与 metadata 校验通过。

## 结果:两次 MCP 派发都在 ~60–70 秒后 `exited_incomplete`

| 运行 | 死亡 epoch | collision@50 | collision@100 | 对照(无该项的上一次) |
|---|---|---|---|---|
| #1 | 149 | 0.8757 | 0.5521 | 0.2154 / 0.2889 |
| #2 | 127 | 0.7713 | 0.5682 | — |

前台复现(带 `-u`)拿到真实异常:

```
assignment_floor.CodebookCollapse: the codeword spacing fell to 0.000e+00
from an initial 4.380e-01 (x0.000) for 50 consecutive steps:
the codebook has collapsed relative to its own resolution
    at model/layers.py:144  verify_separation(...)
```

## 判读

1. 图项确实在起作用:`L_graph` 从 **0.4275** 降到 **0.047**(邻居距离降约 9 倍);
2. 代价是**码本坍缩**:把邻居的编码器输出拉近,等于把所有商品的 `z` 挤到少数区域,量化损失随后把码字也拉到一起,码字间距归零;
3. 早期 collision 已升到 **0.55–0.88**(对照 0.22–0.29),即"提升邻居共享"的路线确实成立,但方式破坏了码本利用率;
4. 防坍缩地板在 ~150 epoch 处按设计中止了运行 —— 这是保护,不是 bug。

## 结论

**按当前形式(无界平方距离 + 10% 份额),该约束不可用**:它换来的邻居聚集以码本坍缩为代价,两个待观察指标里"码本利用率"直接被破坏,share-L1 根本来不及测量。

## 下一步(待定)

**margin 版本(推荐)**:把无界拉力换成有界铰链

```
L_graph^(1) = mean max(0, ‖z_i − z_j‖² − m²) / mean‖z‖²
```

`m` 取**当前各级码字平均间距**(可测量,不是超参),含义是"只把比码本分辨率更远的邻居拉近;已经能被这一级区分的邻居不再施加力"。这样邻居被拉到"刚好可区分"就停,不再把码本压塌。

次选:把约束施加在**分配**上(让邻居落进同一 L1 码字的比例作为软目标),而不是把 `z` 几何拉近 —— 但这会改动 Sinkhorn 的代价矩阵,按规则 19h 属于需要显式批准的研究问题。
