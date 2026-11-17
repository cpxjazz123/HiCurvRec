# Experiment 1：Uniform-KNN 结构保持 Stage2 实验

日期：2026-11-17  
范围：Stage2 单次正式训练 + 离线硬 q 评估；**不运行 Stage3**。

## 1. 实验身份与实现

臂：`TIGER_Uniform_KNN_RQ`  
起点：`stage2_RQ-VAE/baseline/TIGER_RQ-VAE`（只有 `ARM_NAME` 改为本臂）  
训练：3000 epoch，seed 42，三层各 256 码字，32 维潜变量，重建 + quantization 原损失不变。

新增机制：

- 预计算 **sentence-T5 x 空间 K=10 正样本 + 10 个确定性负样本**；
- 固定目标 `P_x`：在 20 个候选上对 cosine 距离做 softmax，温度取该商品正样本平均距离；
- q 侧使用导出 SID 码字和的 **detached q memory** 作为候选；
- anchor 使用 **straight-through soft codebook readout**：
  - 前向仍是硬 q；
  - 反向同时到达 encoder 与 codebook；
- KL 概率地板 `1e-3`，避免重复 q 向量造成数值爆炸；
- 固定 `KNN_WEIGHT=1e-3`，使 KL 项与重建项同量级；不做调参扫描；
- 单次正式训练；不早停、不换种子、不跑消融。

### 训练证据

- smoke：`SMOKE_PASS`，`dead_terms=[]`，`nonfinite_terms=[]`；
- `knn_neighbourhood_kl`、`soft_quantized_readout`、`DISTANCE` 都有非零梯度；
- 最终 epoch 3000：
  - total loss `0.00063788`
  - recon `0.00008845`
  - KNN KL `0.35171`
- 单卡布局，全局 batch 4096；训练墙钟 **1142 秒**。
- 最终硬 SID：21,172 个不同三元组 / 24,587 商品。

## 2. 与实验 0 基线对照

所有 KNN 结果使用同一 K=10、同一对称化并集、同一 Graphlet 标签；q 使用导出 SID 前三码的码字和；z/q 使用 Euclidean，x/x_hat 使用 cosine；q/x_hat 使用 tie-break 集成。

| 指标 | 实验 0 基线 | Uniform-KNN | 变化 |
|---|---:|---:|---:|
| q 真实行为边恢复率 | **1.1279%** | **0.4582%** | **−0.6697 pp / −59.4%** |
| z 行为边恢复率 | 1.7113% | 0.5276% | −1.1837 pp |
| x_hat 行为边恢复率 | 1.1161% | 0.6992% | −0.4169 pp |
| x → z 条件保留 | 65.674% | 9.606% | **−56.07 pp** |
| x → z → q 条件保留 | 28.938% | 7.902% | **−21.04 pp** |
| z → q（给定 x→z） | 44.063% | 82.268% | +38.21 pp |
| q → x_hat（给定 x→z→q） | 87.744% | 64.976% | −22.77 pp |
| q 唯一向量 | 20,425 / 24,587 | 21,172 / 24,587 | +747 |
| q 重复行 | 4,162（16.93%） | 3,415（13.89%） | −747 |
| q K=10 边界并列率 | 22.99% | 14.02% | −8.97 pp |
| L1 已用码字 | 249 / 256 | 216 / 256 | −33 |
| **L2 已用码字** | **256 / 256** | **31 / 256** | **−225（严重坍缩）** |
| L2 有效码字数 | 93.93 | **18.56** | **−75.37** |
| L2 不同前缀数 | 7,374 | **3,307** | −4,067 |
| L3 不同三元组 | 20,425 | 21,172 | +747 |
| L3 distinct collision | 16.93% | 13.89% | −3.04 pp |
| L3 item-weighted collision | 26.53% | 24.53% | −2.00 pp |

**结论：Uniform-KNN 没有改善 q 的行为边恢复，反而把 encoder 和硬 SID 的结构都严重破坏了。**

## 3. 关键异常

### 3.1 L2 码本坍缩

这是本轮最明确的失败信号：

- L2 只用了 **31 / 256** 个码字；
- 有效码字数只有 **18.56**；
- L2 不同前缀从 **7,374** 降到 **3,307**；
- L3 虽然使用了 256 个码字，但无法弥补二级容量损失。

按用户预设的判据，“连续表示改善但 L2 有效容量严重下降、硬 SID 没有改善”不能算成功。本实验比这更差：连续 z 也退化了。

### 3.2 KNN KL 没有下降

- 训练早期 KNN KL：约 **0.3168**
- 最终：**0.3517**
- 期间没有出现稳定下降趋势；
- 同时 z 的向量范数中位数从基线的 **0.145** 膨胀到 **2.815**。

这说明当前软代理优化的对象和硬 q 的邻域结构脱节，且损失把潜空间尺度推大。它不是“KL 起作用但量化没跟上”，而是**训练目标本身没有把 KNN 损失最小化**。

### 3.3 条件保留的分解

- x → z：**9.61%**，编码器几乎丢掉了全部 x 邻域；
- z → q：**82.27%**，量化保住了已经所剩不多的 z 邻域；
- q → x_hat：**64.98%**，decoder 又损失一部分；
- 严格路径 x → z → q：**7.90%**。

与实验 0 的 65.67% / 44.06% / 87.74% 相比，主要失败在 **encoder 阶段**，不是量化阶段。

### 3.4 Graphlet 分型

| 类型 | n | 基线 q | Uniform q | 基线 x→q 条件 | Uniform x→q 条件 |
|---|---:|---:|---:|---:|---:|
| triangle | 30,437 | 3.253% | 1.249% | 35.68% | 9.03% |
| star | 35,139 | 0.542% | 0.192% | 26.997% | 4.13% |
| chain | 13,420 | 0.328% | 0.253% | 25.00% | 11.00% |
| cycle | 24,935 | 1.193% | 0.450% | 33.10% | 8.72% |
| mixed | 42,539 | 0.920% | 0.347% | 34.43% | 8.03% |
| low | 168,921 | 0.973% | 0.417% | 31.41% | 8.35% |

所有类型都退化；没有出现“某类结构被 Uniform-KNN 改善”的证据。

## 4. 判定

**本轮 Uniform-KNN 失败。**

不是因为“q 没改善但 z 改善了”，而是：

1. z 和 q 都显著退化；
2. L2 有效容量坍缩到 18.56；
3. KNN KL 本身没有下降；
4. 硬 SID 的层级容量从 7,374 个 L2 前缀降到 3,307。

因此：

- **不能**把这次结果解释为“普通 KNN KL 无效”；
- 更准确的解释是：**当前是可微代理/尺度控制失败的 Uniform-KNN 实现**；
- 也不能进入 C（Graphlet-KNN）或 D/E；在 Uniform-KNN 修复并重新验证前，Graphlet 选择性没有可比基线。

## 5. 下一步建议

1. **先修 Uniform-KNN，不要直接做 C**：
   - 对 z/q 做显式范数约束或改用归一化 cosine KL；
   - 去掉或修正 q memory 的 stale 候选路径；
   - 加入可监控的码本使用/熵约束，至少防止 L2 坍缩；
   - 让 KL 在训练中真实下降，再比较硬 q。
2. **保持一轮一个机制、单次训练**；修复后用新的候选树重新跑，不覆盖本次臂。
3. **本轮不启动 Stage3**：Stage3 的硬碰撞门槛虽然通过（distinct collision 0.1389 < 0.40），但 L2 坍缩和结构退化使 Recall 比较没有意义。

## 6. 产物

- 候选树：`stage2_RQ-VAE/TIGER_Uniform_KNN_RQ`
- 训练 SID：`results/stage2_RQ-VAE/TIGER_Uniform_KNN_RQ/item_sids.json`
- checkpoint：`results/stage2_RQ-VAE/TIGER_Uniform_KNN_RQ/rqvae_best.pth`
- 行为恢复：`results/research/uniform_knn_evaluation/uniform_knn_metrics.json`
- 原始表格：`uniform_knn_metrics.md`
- SID 使用率：`uniform_knn_sid_usage.json`
- Stage2 分层指标：`results/stage2_RQ-VAE/TIGER_Uniform_KNN_RQ/stage2_metrics/e460a39b02f14c9653fcab1e4b112b780727e25dda1576674d50f6145dbba36f.json`

本报告没有使用 Stage3 结果，也没有把测试集用于训练或选择。
