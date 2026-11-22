## round 1 正式训练启动:三组对照(并行,§19d0c)

### 三组

| # | 臂 | 树 / 数据 | GPU |
|---|---|---|---|
| 1 | 原始基线 256×3 | `results/stage2_RQ-VAE/TIGER_RQ-VAE`(已核验,复用) | — |
| 2 | 新硬 SID Loss | `stage2_RQ-VAE/TIGER_sid_graphlet_rq` + `issue231/banks` | 0 |
| 3 | 真角色打乱对照 | `stage2_RQ-VAE/TIGER_sid_graphlet_shuffled_rq` + `issue231/shuffled_banks` | 1 |

### 对照是否真的不同(C38)

`shuffled_banks` 由 `role_shuffle.shuffle_roles` 生成,**改变率 1.0 × 三结构**(逐个验证标签不同):

| 结构 | 实例 | 改变率 |
|---|---:|---:|
| chain | 1,138 | 1.0 |
| fork | 1,059 | 1.0 |
| merge | 1,189 | 1.0 |

两棵树只差两行:`ARM_NAME` 与 `GRAPHLET_BANKS`。validation/test 完全共用,所以评估集一致。

### 固定项(§1/§19g)

seed 42、3000 epoch、全局 batch 4096、码本 256×3、`GRAPHLET_SHARE=0.10`、
结构与权重均**未按结果调整**。

### 下一步

跑满后按 §5 在导出硬 SID 上复算 C40–C51(含跨结构隔离的冻结 test bank,各约 150 例,
区间会宽 —— 按 C44 如实报告宽度)。
