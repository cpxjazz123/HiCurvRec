## C25/C29/C32:机制已实现并通过小规模测试,1-epoch 真实入口 smoke 通过

### 机制(§3)

`losses/hard_prefix.py`,接入点是 1 级 cost 矩阵,在 Sinkhorn/floor 分配**之前**相加(§19h:分配规则不动,机制进入代价矩阵)。

- **前向是硬的**:`argmin` 取真实码字,`S()` 与导出件完全一致(0..1,只到 1 级);
- **反向是 STE**:门控由硬 gap 决定,斜率由 soft 代理给出;
- **温度由数据推导**:实测 1 级 cost 均值 6.5e-2、最近/次近码字间距中位 4.2e-3,而 `softmax(-cost)` 的熵是 6.93 ≈ ln(1024)(完全均匀,梯度 2.7e-11,死掉)。取中位间距的一半(2.1e-3)后熵 1.86、每商品约 10 个候选码,梯度均值升到 **8.0e-6**。

### 为什么改在 1 级

`COMMENT_11` 的三层分解:基线 Δ 的 87.5%(chain)、86.4%(fork)、92.3%(merge)在 **L1** 就已经建立,L2/L3 同号且小一个量级。所以只监督 1 级,不在 2/3 级重复一遍。

### 测试(7 个,先红后绿)

| 测试 | 验证 |
|---|---|
| `test_hard_forward_reports_the_true_prefix_match` | 前向值与 `S()` 手算一致(chain/fork 各一例) |
| `test_hard_forward_disagrees_with_its_own_soft_surrogate` | 前向不是软代理(数值不同) |
| `test_gradient_reaches_the_logits_and_points_the_right_way` | **一步 SGD 后共享率上升、端点对共享率下降** |
| `test_the_term_reorders_a_near_tie_in_the_real_assignment` | C29:微扰改变真实 `argmin`,不只是概率 |
| `test_role_index_reads_the_contract_banks` | 训练侧读冻结 bank |
| `test_role_index_rejects_an_evaluation_pair_in_the_loss` | C9:运行期隔离断言 |
| `test_role_index_refuses_a_bank_whose_wiring_disagrees` | C30:bank hash 不符即抛错 |

`pytest tests/` → **11 passed**(含原有 4 个 wide-L1 测试)。

### 真实分配是否改变(不是只在软概率上)

同一次前向,1024 码 L1:

| 量 | 值 |
|---|---|
| `_prefix_adjust` 输出绝对值均值 / 最大 | 8.0e-6 / 3.79 |
| 该 batch code 翻转数 | **93 / 4096** |

### 1-epoch smoke(真实入口,4 卡)

| 项 | 值 |
|---|---|
| 命令 | `ISSUE230_SMOKE_EPOCHS=1 python3.9 train_rqvae.py` |
| 输出目录 | `results/research/issue230/smoke/TIGER_graphlet_wide_l1_rq/`(不污染正式结果目录) |
| world_size | 4 |
| epoch 1 | loss=8.96e-4, recon=4.72e-4 |
| 碰撞率 | 0.021922(raw_unique 24048/24587) |
| 导出 | `sid_shape=(24587,4)`,unique 24587 |
| 耗时 | 65s |

1 epoch 的 Δ(chain +0.0045 / fork −0.0127 / merge −0.0036)与基线不可比 —— 基线要跑满 3000 epoch,

### 下一步

按 §19g 起正式 3000-epoch 单次运行(seed 42、全局 batch 4096),完成后用 `COMMENT` 里的同一口径在冻结 test bank 上算 C36–C41。

证据:`stage2_RQ-VAE/TIGER_graphlet_wide_l1_rq/losses/hard_prefix.py`、`tests/test_hard_prefix.py`、`results/research/issue230/smoke/`。
