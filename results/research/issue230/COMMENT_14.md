## CHECK_2:训练臂配置对账通过(C31/C33/C34/C35),并保住了一个差点被覆盖的对照臂

### 本轮判定

| 编号 | 判定 | 证据 |
|---|---|---|
| C31 | **PASS** | `config/train_config.json`:`EPOCHS=3000`、`SEED=42`、`CODEBOOK_SIZE=[1024,256,256]`,与基线逐项一致;全局 batch 4096 |
| C32 | **PASS** | 治疗臂 7 个测试、对照臂 16 个测试全绿;1-epoch smoke 导出 SID、碰撞 0.021922 |
| C33 | **PASS** | 基线复用(C12 已核验);旧 soft-loss 臂已快照保存(见下);治疗臂与打乱臂在训 |
| C34 | **PASS** | 两臂读同一 `results/research/issue230/banks`,`RoleIndex` 对 `banks.json` 做哈希断言 |
| C35 | **PASS** | 没有权重搜索:`MARGIN=1.0`、`BUDGET=0.10` 固定,温度由数据推导(最近/次近码字间距中位数的一半) |
| C36–C44 | **NOT_RUN** | 需要 3000-epoch 导出的 SID,两臂仍在训练 |

### 发现并修掉的问题:治疗臂会覆盖旧的 soft-loss 对照

治疗臂的 `ARM_NAME` 是 `TIGER_graphlet_wide_l1_rq`,而该目录下的 `item_sids.json` / `rqvae_best.pth`
**就是旧联合 soft loss 臂的产物**(mtime 11:57,早于本次训练)。若不管,训练结束会把它覆盖,C33 的第三组
"旧联合 soft loss" 就永久丢失。

处置:训练结束前已把三件产物快照到 `results/research/issue230/old_soft_loss/`,

| 文件 | sha256 |
|---|---|
| `item_sids.json` | `538fb141a4b19b689cf6abc8493d76f71128403482a2b129ad0d47b6a73f7b2f` |
| `sids_for_hgrec.npy` | 已保留 |

该 sha256 与 `COMMENT_5` 里 C12 复核时的记录一致,说明快照就是当时那份,没有混入新产物。

### 训练状态

| 臂 | epoch | 进程 cwd |
|---|---|---|
| 新硬前缀 | 2033/3000 | `stage2_RQ-VAE/TIGER_graphlet_wide_l1_rq` |
| 打乱对照 | 1809/3000 | `stage2_RQ-VAE/TIGER_graphlet_prefix_shuffled_rq` |

两臂各占一张卡(GPU 0 / GPU 1),互不干扰;按 §19d0b 并行派发。

证据:`results/research/issue230/CHECK_2.json`、`results/research/issue230/old_soft_loss/`。
