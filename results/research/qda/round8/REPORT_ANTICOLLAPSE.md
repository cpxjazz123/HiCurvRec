# 通用防坍缩层:任一 Stage2 机制都能继承的分配约束

**结论**:把"分配"这一步换成"保持每个码字至少 1 个商品的最近邻分配",任何机制都不可能再让码字饿死——因为死亡的机制本身就是"该码字在这一步没有商品",而这个分配规则从构造上排除了它。两个臂实测:**训练期每一 epoch、每一级的活跃码恒为 256/256**,并且关掉 Sinkhorn 的版本让训练与导出**逐项一致**。

## 1. 机制:只换分配,不动别的

任何 Stage2 机制最终只给量化器一样东西:**代价矩阵 `(items, codewords)`**。这一层就作用在它上面:

```python
# model/mixed_rq.py
ASSIGNMENT_FLOOR = 1                      # 每个码字至少 1 个商品
FORCE_CONSTRAINED_ALL = True              # 关掉 Sinkhorn,训练与导出共用一条规则

# _indices(训练)与 encode_tokens(导出)里的 argmin 都被替换成:
index, _ = constrained_assignment(cost, ASSIGNMENT_FLOOR)
```

`constrained_assignment` 做的是"带下限的最近邻":先取 argmin,然后每一轮把所有未达下限的码字**一次性**各自抢走一个最便宜的、来自有盈余码字的商品(一波矩阵运算,不是逐码循环);重复直到全部达标。每次搬运至少消掉一个缺口 ⇒ 必然终止;`items < codewords × floor` 时**抛 `InfeasibleFloor`**,不返回"部分满足"。

**为什么这是"任意机制"都能用的**:它不吃损失、不吃距离、不吃曲率、不改码本参数化、不改编码器。候选只要用 `MixedVQLevel`,复制这三处改动即可;机制自己怎么写,与这层无关。上界成本也测过:

| 情形 | 每次调用 | 搬运数 |
|---|---:|---:|
| 已坍缩(8 个码活着) | 45 ms | 248 |
| 健康(每码已被自然选中 ≥7 个) | **0.43 ms** | **0** |

也就是说收敛之后这层**是空转的**:自然分配本来就满足下限,它不改变任何商品的归属。

## 2. 实测:码字不再死亡,训练与导出可以合一

| 臂 | 配置 | 训练期活跃码(逐 epoch 最小/最大) | 训练=导出 | L3 collision |
|---|---|---|---|---:|
| F0(对照) | argmin;L3 Sinkhorn | L1 8–202 | L1 1.0000 / L3 0.796 | 0.3447 |
| **H1_floor_argmin** | argmin 路径加下限,L3 仍 Sinkhorn | **L1 256–256,L2 256–256**;L3 230–256 | L1/L2 1.0000,**L3 0.4792** | **0.0834** |
| **H2_floor_all** | 下限作用于三级,Sinkhorn 关闭 | **L1/L2/L3 全部 256–256** | **1.0000 / 1.0000 / 1.0000** | 0.1098 |

- **H1** 证明下限本身有效:L1/L2 从第一个 epoch 起就是 256/256(基线在 epoch 8 只剩 25)。
- **H1 的 L3 只有 0.4792 的训练/导出一致率**,因为它的 L3 仍在训练期走 Sinkhorn、导出走带下限的最近邻 —— 这正好说明"只改训练侧"是不够的。
- **H2** 把 Sinkhorn 关掉、三级都用带下限的最近邻:训练与导出**逐项一致**,并且每一级、每一 epoch 都是 256/256。**这才是完整形态**。

顺带一个必须记录的细节:观察器里的 `usage_*` 是它自己算的**纯 argmin**,不是训练用的规则,所以 H1 的 `usage_2` 会出现 0 —— 那正是"如果不把下限写进导出规则,导出侧仍会出现空码"的证据。

## 3. 价格:占用拉满,但表示仍掉一点

300 epoch,同判据:

| 臂 | L1/L2/L3 码 | encoder | L1 AUC | L2 | L3 AUC | L1 缺口 | collision |
|---|---|---|---:|---:|---:|---:|---:|
| F0 | 26/153/256 | **0.6594** | 0.6459 | 0.6547 | **0.6592** | +1.35 pp | 0.3447 |
| B3(L1 Sinkhorn) | 256/124/253 | 0.6596 | **0.6598** | 0.6585 | 0.6579 | −0.02 pp | 0.0935 |
| H1 | 256/256/245 | 0.6565 | 0.6508 | 0.6540 | 0.6553 | +0.57 pp | **0.0834** |
| H2 | 256/256/256 | 0.6545 | 0.6513 | 0.6541 | 0.6538 | **+0.32 pp** | 0.1098 |

- **唯一性最好**:H1 的 collision 0.0834 是全系列最低(比 B3 的 0.0935 还低)。
- **量化缺口降到 1/4**(+1.35 → +0.32 pp),但编码器掉 0.5 pp,于是 L3 行为 AUC 仍低 0.54 pp。
- 这与前六轮完全一致:**L3 的天花板不由 L1 占用决定**。这一层买的是"不会交付坍缩模型",不是"更高的 L3"。

## 4. 怎么套到任意新机制上

1. 复制 `qda_floor.py`(60 行,只依赖 torch)到候选树;
2. 在 `model/mixed_rq.py` 里加 `ASSIGNMENT_FLOOR` / `FORCE_CONSTRAINED_ALL` 两个常量;
3. 把 `_indices` 与 `encode_tokens` 里的 `argmin` 换成 `constrained_assignment(cost, ASSIGNMENT_FLOOR)`;
4. 跑 `round7/certify_existing.py` 的闸门:容量、几何、有效秩、行为保真四条全过才允许交付。

候选自己的损失、距离、曲率、码本初始化一律不动。可行性只有一条:`batch ≥ codewords × floor`(4096 ≥ 256×1,余量 16 倍)。

## 5. 边界(必须同时写)

- 这层保证的是**占用**,不是**质量**。退化编码器(所有商品同一向量)照样能满足下限 —— `round7` 的对抗例已经证明这种情况必须由认证层的有效秩与行为保真拒绝。
- 它不提高 L3:本轮 H1/H2 的 L3 仍略低于 F0。**防坍缩 ≠ 提升结构保真**,两者在这个模型上是分开的。
- `floor` 取 1 是"最小可证"的选择:它足以让吸收态不可达;取更大值会强制更多商品去不合适的码字,本轮的 `moves` 统计(收敛后为 0)说明 floor=1 在健康阶段完全空转,而更大 floor 会持续搬运。

## 6. 复现

```bash
<genrec_env_v2>/python3.9 results/research/qda/round8/build_h_arms.py
cd results/research/qda/round8/trees/H2_floor_all && <genrec_env_v2>/python3.9 train_rqvae.py
<genrec_env_v2>/python3.9 results/research/qda/round2/analyze_round2.py \
    --runs results/research/qda/round8/runs --arms H1_floor_argmin H2_floor_all --no-strata \
    --out "$PWD/results/research/qda/round8/report_h.json"
```

产物:`floor_src.py`(安装为各树的 `qda_floor.py`)、`build_h_arms.py`、`arm_manifest.json`(逐文件 sha256)、`runs/{H1,H2}/{logs,qda_epochs.json,observations}`、`report_h.json`。
