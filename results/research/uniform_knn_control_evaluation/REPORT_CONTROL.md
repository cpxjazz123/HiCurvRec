# Strict argmin no-KL Stage2 control and final comparison

日期：2026-11-18  
范围：严格匹配的 Stage2 对照 + 离线精确 q 评估；**未运行 Stage3**。

## 1. 对照身份

`TIGER_Uniform_KNN_control` 由当前 Uniform-KNN 树复制，**只把 `KNN_WEIGHT` 设为 0.0**：

- 同一 argmin 分配、同一单卡全局 batch 4096、同一 seed/初始化/训练预算；
- 同一 q memory 代码，只是 KL 不进入梯度；
- 同一 checkpoint-bound CPU export；
- 3000 epoch，**1104 秒**；
- final collision **0.18315**；
- export `tokens_match=true`（checkpoint-bound CPU）。

## 2. 三组对照（均为导出 SID 查码本生成的精确 q）

| 指标 | 纯欧氏基线 | argmin 无 KL 对照 | argmin + Uniform-KL |
|---|---:|---:|---:|
| q 行为边恢复率 | 1.1279% | **1.1271%** | **0.9376%** |
| z 行为边恢复率 | 1.7113% | 1.6988% | 1.6151% |
| x_hat 行为边恢复率 | 1.1169% | 1.1162% | 0.9487% |
| x → z 条件保留 | 65.674% | **66.288%** | 60.787% |
| x → z → q 条件保留 | 28.938% | **28.477%** | 21.605% |
| z → q（给定 x→z） | 44.063% | 42.960% | 35.542% |
| q → x_hat（给定 x→z→q） | 87.784% | 87.601% | 85.263% |
| q 唯一向量 | 20,425 | **20,084** | 15,338 |
| q 重复行 | 4,162 | **4,503** | **9,249** |
| q K=10 并列率 | 22.99% | 24.63% | **48.29%** |
| L1 已用码字 | 249 | **244** | **114** |
| L2 已用码字 | 256 | **256** | 256 |
| L3 已用码字 | 256 | **256** | **69** |
| L2 不同前缀 | 7,374 | **8,199** | 4,988 |
| L3 distinct collision | 16.93% | **18.31%** | **37.62%** |
| L3 item-weighted collision | 26.53% | **28.88%** | **53.98%** |

## 3. 结论

严格对照说明：

1. **argmin + 单卡本身几乎完全复现基线**：q 1.1271% vs 1.1279%，L1/L2/L3 码本都健康。
2. **Uniform-KL 的真实贡献是负的**：
   - q 0.9376% vs 对照 1.1271%（相对下降约 16.8%）；
   - x→z→q 21.605% vs 28.477%；
   - L1 码字 114 vs 244，L3 码字 69 vs 256；
   - L3 distinct collision 37.62% vs 18.31%。
3. 因此，之前无法归因的 **L1/L3 坍缩不是 argmin 或单卡造成的**，而是 Uniform-KL 训练目标本身造成的。
4. 这轮可以判定：**当前 Uniform-KNN 实现没有改善硬量化结构，且严重损害码本容量**。

## 4. 边界

- 只做一次对照，没有调参或多种子；
- 没有运行 Stage3；
- 不能把 `technically_valid_sid=false` 的 Uniform 臂拿去与 Stage3 比较；
- 控制臂 `stage2_metrics` 的 `technically_valid_sid=true`。
