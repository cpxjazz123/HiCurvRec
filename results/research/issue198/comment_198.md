## 完成：A/B/C 全部交付，D 三条假设逐条定状态；**不另开修复 Issue**

产物：`results/research/issue198/`（`audit_a.py` / `audit_b.py` / `audit_c.py` + `audit_a.json` / `audit_b.json` / `audit_c.json` + `A_RECIPE.md` / `B_GRADIENT_SCOPE.md` / `C_D_VERDICT.md`）。**未训练、未派发、未产生新 SID。**

### A 配方一致性 ⇒ **`EXCLUDED`**

两树常量**逐项相同**（3000 epoch、1024/卡、AdamW lr 1e-3 wd 1e-4、warmup 50、seed 42、3×256×32、Sinkhorn 0.003/50、无 PCA、clip 1.0）。SIGN 的 `train_start` 实测：`world_size=4`、`batch_size_per_rank=1024`、**`total_batch_size=4096`**、`steps_per_epoch = ceil(24556/4096) = 6`，与"每卡 6139 件 ÷ 1024"一致。**#181 的 16,384 事故在此不存在。** 基线侧 `TIGER_mixed_rq` 无 metrics 文件，其 logged 配方记为 `NOT_AVAILABLE`，可比的是源码常量（全同）。

### B 曲率作用范围 ⇒ **`EXCLUDED`**

- 路由是**唯一一处显式表达式**：`mixed_rq.py:377` 与 `:397` 的 `kappa=(kappa if level == 1 else None)`；二三级恒 `None`。与 #193 的"通用 `VQLayer` 模块级挂钩"**不是同一形态**。
- 真实 checkpoint + 真实 batch 1024 的分块梯度范数（门开 / 门关）：encoder **2.7e-05**、codebook **1.1e-05**、decoder **0**（相对差）。
- 28 件的 κ：min 0.0、q25 0.0294、**median 0.0959**、q75 0.3883、max 8.2671、mean 0.5351 —— **多数启用件的曲率弱于基线**。
- 每 batch 命中：均值 **1.1**、最大 3、**6/20 个 batch 为 0**（期望 1.168）；`kappa_rows_uniform_one_share = 0.99886`。

### C 选码与坍缩对应

- **同一 checkpoint：门开 vs 门关，24,587 件逐级差异 L1/L2/L3 = `0 / 0 / 0`**，三码全同 **100%**；单 batch（含 4 件启用）L1 改变 **0** ⇒ **推理期无传播**。
- 一级直方图：SIGN **48 码 / top1 11.37% / top5 39.52% / 熵 83.67%**；`TIGER_RQ-VAE` 247 / 3.49% / 13.10% / 86.91%。

### **关键：参照系选错了（这是本次审计最重要的发现）**

SIGN 的起点是 `TIGER_mixed_rq`（E16×H16 联合码本 + α 混合），**不是** `TIGER_RQ-VAE`（E32 欧氏）。同族对照（各 24,587 件、同一门算式）：

| 臂 | 门 `collision` | 二级前缀 | 不同三码 | 一级码数 | 一级 top1 |
|---|---|---|---|---|---|
| `TIGER_mixed_rq_F0`（同 E16×H16、α=1/2） | 0.257168 | 4,151 | 18,264 | **79** | 5.89% |
| `TIGER_M1_learned_kappa`（同族） | 0.315329 | 2,933 | 16,834 | **42** | 14.06% |
| **SIGN** | 0.308090 | 3,081 | 17,012 | **48** | **11.37%** |
| `TIGER_RQ-VAE`（E32 欧氏，Issue 参照列） | 0.177492 | 7,414 | 20,223 | 247 | 3.49% |

⇒ **"一级 48 码"是共享联合码本这一结构的性质**（F0 同配方下也只有 79 码 / 25.7% 碰撞），SIGN 落在自己家族区间内。

### D 归因

| # | 假设 | 状态 |
|---|---|---|
| (i) | 配方 / batch 不一致 | **`EXCLUDED`**（A） |
| (ii) | 曲率作用范围泄漏 | **`EXCLUDED`**（B：路由显式、梯度差 1e-5、三层选码差 0） |
| (iii) | 28 件经码本共享传播改变全局码 | 推理期 **`EXCLUDED`**（0/24,587）；训练轨迹差异 **`INSUFFICIENT`**（每步梯度差 ~1e-5 × 3000 epoch 可累积成不同轨迹，但**不新增第二臂无法分离**，Owner 已禁消融/重跑） |

### 结论

**"28 件 ⇒ 48 码"的因果链不成立**：机制在发布的 checkpoint 上不改变任何一件商品的码；48 码是该量化器家族的水平。因 (i)/(ii) 均已排除且 (iii) 推理期排除，按本 Issue 规定**不另开单臂修复 Issue**。`#197` 的 `test_R@10 = 0.05609429` 仅并列写入，不作归因依据。

**依据 commit**：`1ff0bcda2`（A）· `1e2c9e909`（B）· `09859e190`（C+D 与判定）。
