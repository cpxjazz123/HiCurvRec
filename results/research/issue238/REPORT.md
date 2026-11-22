# 探索轮:把图结构表示接进 RQ-VAE 的两条路径

owner 指示:自由探索不同方向/机制,完成 Stage2 + Stage3 并拿到结果。

## 1. 为什么选这条线(不重复 #234)

| 已有证据 | 结论 |
|---|---|
| #234 六个 λ 全部无质量合格点,Stage3 recall 随 λ **单调下降**(0.0530→0.0431) | 「在 RQ-VAE 目标里加结构 loss」已证伪 |
| #234 的 graphlet 对齐指标改善,但 Stage3 变差 | 对齐指标**不是** Stage3 收益的代理 |
| #235/#236 曲率 −1.02、符号由度数决定 | 图负曲率为主但异质 |
| #237 图传播表示:链接预测 AUC 0.596→**0.656**,语义 cosine 0.005 | 存在纯结构 768 维表示,**但 #237 明确把「是否接入 RQ-VAE」留给后续** |

因此本轮把变量放在**输入表示**上,不动目标函数,避开 #234 那条「结构 loss 与重建竞争」的失效路径。

## 2. 两条臂(每臂只改 2 行)

| 臂 | 输入 | 与冻结基线的 diff |
|---|---|---|
| `TIGER_g237_struct_rq` | #237 variant A(纯结构,cos 0.005) | `ARM_NAME` + `EMBEDDING_FILE` |
| `TIGER_g237_sem_rq` | #237 variant B(语义保真,cos 0.881) | `ARM_NAME` + `EMBEDDING_FILE` |

RQ-VAE 结构/损失/优化器/Sinkhorn 分配/导出协议/3000 epoch/seed 42 与基线逐字节一致。

## 3. 结果

### 3.1 Stage2(四卡 3000 epoch,均跑满)

| 臂 | 训练器导出的 collision | checkpoint 重导出的 collision | 最大桶 | 与训练器表不同的商品数 |
|---|---:|---:|---:|---:|
| TIGER 基线 | 0.1693 | **0.1693**(122–291 项差异) | — | 122–291 |
| **G1 struct** | **0.0036** | **0.9655** | **13,810** | **24,587(全部)** |
| G2 sem | 0.0092 | 0.1738 | 71 | 24,575 |

### 3.2 Stage3

| 臂 | 结果 |
|---|---|
| **G1 struct** | **拒绝启动**:`SID quality gate: three-level collision 0.965470 is not below 0.40` |
| G2 sem | 见 §3.3 |

## 4. 结论:两条路径都失败,但失败方式不同

### 4.1 G1:checkpoint 不可复现地坍缩

训练器内存里导出的是 **24,498 个互不相同的三码**(collision 0.0036);
而 `stage3_start` 按 CLAUDE.md §67 的 export_gate 从 checkpoint 重导,得到
**849 个三码、最大桶 13,810 个商品**、collision **0.9655**。

`export_manifest.json` 记着 `original_differing_items: 24587` —— **全部商品都变了**。

即:**#237 的纯结构表示可以让 RQ-VAE 在训练时看起来完美区分(0.0036),但保存下来的
checkpoint 前向会把 56% 的商品塞进同一个码字。** 这不是抽样噪声,是分配在该表示下
不可复现。基线在同样协议下只差 122–291 项(0.3–1.2%)。

### 4.2 G2:同样大幅漂移,但侥幸低于门槛

训练器 0.0092 → checkpoint 0.1738(24,575/24,587 项不同),刚好低于 0.4 硬门槛,
因此 G2 得以进入 Stage3。**它与基线 0.1693 已经几乎同一量级**,说明 G2 的 SID 在
重导之后回到了基线那种「大量共享码字」的形态。

### 4.3 机制层面的教训

1. **「训练时 collision 低」不能作为 SID 质量的证据**:两个臂的训练器值(0.0036/0.0092)
   与 checkpoint 值(0.9655/0.1738)相差两个数量级。CLAUDE.md §67 的 export_gate 正是
   为此存在,这次它拦下了 G1。
2. **#237 的表示不适合直接作为 RQ-VAE 输入**:它的 latent 分布让码字分配对
   checkpoint/内存差异极度敏感。若要接入,需要先解决量化稳定性(例如在 Stage2 目标里
   加码本利用率约束,或改用 #237 的切空间输出做 PCA 后再量化)。
3. 这条结论**不是**「混合曲率无效」:G1/G2 只是把 #237 的表示喂给标准 RQ-VAE,
   没有改动几何或分配规则。

## 5. 一个必须先修的框架缺陷(已修)

首次启动任何新臂都失败:

```
UnknownTree: model/layers.py: anchor appears 0 times, expected 1:
    embed_ind = self._indices(cost, infer_use_sk)
```

冻结基线的 `VQLayer.forward` 用内联 `self._indices(self._distances(latent), …)`,
而 `apply_floor` 的编辑清单只覆盖「提升为 `cost`」(#232 风格)与「EMA」两种形态,
**缺了基线本身这一种**。也就是说:按 §19b 从冻结基线复制的新臂**一律无法启动**。

已修 `tools/research/stage2/anticollapse/patch_tree.py`:
- 补上基线形态的锚点;
- 两种树变体专属的编辑标为可选(`unguarded_sites` 仍是安全网);
- 回归测试 `tools/research/stage2/tests/test_floor_install_on_baseline.py`
  (复制冻结基线 → 安装 floor → 断言无未保护赋值 + 幂等)。

## 6. 复现

```bash
PY=/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9
cd /fs04/ar57/wenyu/GeneRec
$PY results/research/issue238_run_g1.py    # Stage2 G1(四卡 3000 epoch)
$PY results/research/issue238_run_g2.py    # Stage2 G2
$PY results/research/issue238/run_stage3.py  # 两个 Stage3 串行
```

## 7. 证据哈希

| 臂 | 表 | sha256(前16) | 字节 |
|---|---|---|---|
| TIGER_g237_struct_rq | item_sids.json | b76cd843b70cce01 | 1317553 |
| TIGER_g237_struct_rq | item_sids.trainer.json | d41a8877b7ca5a18 | 1259084 |
| TIGER_g237_sem_rq | item_sids.json | e992b716bbf95d68 | 1258845 |
| TIGER_g237_sem_rq | item_sids.trainer.json | 60dfafcefb8ece11 | 1261237 |

## 8. 第三条机制:测地代价进 Sinkhorn(H,`TIGER_kappa_cost_rq`)

### 为什么是这里

§19h 规定机制创新必须进 Sinkhorn 的代价矩阵,分配规则仍是基线的均衡分配;`distances/__init__.py`
自己的约定是「换度量 = 换实验,选择写在源码里」。所以 H 的改动只有三处:

| 文件 | 改动 |
|---|---|
| `distances/curved.py` | 新增:κ=−1 球面平方测地距离 |
| `distances/__init__.py` | 注册 `"curved"` |
| `train_rqvae.py` | `build_distance(CONFIG, name="curved")` |

输入嵌入、损失、训练超参全部未动,所以量化稳定性应当接近 TIGER 基线(与 G1 的 0.9655 崩塌对比)。

### 定义

```
u = 0.9 · tanh(r/τ) · x/r ,  r=|x| ,  τ = mean‖c‖(当前 step)
d(u,v)² = ( arcosh(1 + 2‖u−v‖² / ((1−‖u‖²)(1−‖v‖²))) )²        (κ = −1)
```

- κ 取 #236 实测 −1.0188 的整数点 −1,**不是搜索变量**(§19g 不做调参)。
- τ 取当前码本平均范数,使映射尺度等变,**没有自由超参**。
- 半径上界 0.9 是数值条件数要求,不是拟合参数:不加时 `(1−‖u‖²)` 极小,展开式
  `‖u−v‖²` 的浮点误差被放大,自距离实测 **14 而非 0**。这是本轮真实踩到的坑,已写进模块注释。

### 数学验证(`results/research/issue238/tests/test_curved_cost.py`,7/7 通过)

| 性质 | 结果 |
|---|---|
| 映射后半径 < 1 | 通过(50× 放大输入仍封在球内) |
| 自距离 = 0 | 通过(≤1e-5) |
| 非负、有限、形状 (B,K) | 通过 |
| 沿穿过码字的射线单峰(近侧单调降、远侧单调升) | 通过 |
| 两个点集间对称 | 通过 |
| 与欧氏代价不同(机制未退化为基线) | 通过 |
| 梯度流向 latent 与 codebook | 通过 |

**为什么可能有收益**:负曲率球上靠近边界的点距离增长更快,把紧簇拆到不同码字的代价高于欧氏;
Sinkhorn 仍保证均衡(§19h 的分配规则不变)。
