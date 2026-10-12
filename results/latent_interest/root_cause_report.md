# 定位 RQ-VAE 三层行为信息丢失的根本原因

仓库 `cpxjazz123/HiCurvRec`，承接 Issue #2。本轮起点 `c48e903`。
**结论先行：行为信息不是在「残差形成」或「量化」阶段丢失的，而是在梯度层面从未到达 L2/L3。**

---

## 0. 三个问题的答案

| 问题 | 答案 | 决定性证据 |
|---|---|---|
| **行为信息在何处丢失？** | **在梯度通路**：L2/L3 残差对所有可训练参数是 detached 常数，其对比损失梯度**恰为零** | 实测 `\|encoder grad\|_1` = 30.48（L1）vs **0.0**（L2/L3）；恒等式 `r1 ≡ −(quant − raw).detach()` 差 `0.0`；`l2only`/`l3only` 跑满 3000 epoch 后 checkpoint **逐字节等同**无监督基准 |
| **是否值得在 L2/L3 施加行为监督？** | **不值得——按当前残差构造是可证明的 no-op** | 同上；要把深层监督变成有效，必须换作用点（码本/分配），而不是换残差 |
| **上一轮的 Q2（量化不是瓶颈）还成立吗？** | **仍成立，但理由必须改** | 修正后的定义下：连续排对后量化为 **Tie** 79.9%/97.2%/99.5%，**反转**仅 0.79%/0.03%/0.00% |

---

## 1. 根因：straight-through 使深层残差梯度隔离

`stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/model/layers.py`（`VQLayer` 与 `EMAVQLayer` 同一模式）：

```python
quant_loss = F.mse_loss(x_q, x.detach()) + self.beta * F.mse_loss(x, x_q.detach())
x_q = x + (x_q - x).detach()        # 数值是 x_q，梯度等同于 x
```

代入 `_residuals_with_trace` 的残差递推（`level0 = encoder(x)`，`level_{l+1} = level_l − quant(level_l)`）：

$$r_{l+1} \;=\; r_l - \bigl(r_l + (q^{code}-r_l).\mathrm{detach}\bigr) \;=\; -\,\bigl(q^{code}-r_l\bigr).\mathrm{detach}$$

⇒ **L2/L3 残差是 detached 常数**，对比损失在这两层对每个可训练参数的梯度**恰为零**。

### 两条独立证据闭合

**（a）数值实测**（真实批次、真实配对）：

| 层 | 损失 | `\|encoder grad\|_1` | 参数缺梯 |
|---|---|---|---|
| L1 | 4.1577 | **3.047568e+01** | 0/8 |
| L2 | 4.1597 | **0.000000e+00** | 0/8 |
| L3 | 4.1593 | **0.000000e+00** | 0/8 |

代数恒等式校验：`r1 == -(quant − raw).detach()`，**max diff = 0.0**。

**（b）端到端**：`resample_l2only` / `resample_l3only` 各跑满 3000 epoch（损失全程活跃：6.31 / 6.30，weight 0.2），产出的 checkpoint 与 SID **逐字节等同于无监督基准**：

```
018ef7b19d471714…  TIGER_RQ-VAE/rqvae_best.pth
018ef7b19d471714…  resample_l2only/rqvae_best.pth
018ef7b19d471714…  resample_l3only/rqvae_best.pth
471b8511e9ec95f5…  TIGER_RQ-VAE/sids_for_hgrec.npy
471b8511e9ec95f5…  resample_l2only/sids_for_hgrec.npy
471b8511e9ec95f5…  resample_l3only/sids_for_hgrec.npy
```

24587 个商品、前三个 token 差异 = **0**；三层 AUC 与基准逐位相同。

### 对既有机制描述的实质性更正

L2/L3 项**逐参数梯度为零**，但其**常量值参与了 `torch.stack(losses).mean()`**，因此把 L1 的有效梯度整体缩小 **1/3**。所以准确表述是：

> **Exp1a 不是三层残差监督，而是 1/3 强度的 L1 监督。**

确认：`l1only`（全强度 L1）在三层上都与 Exp1a 统计不可区分（见 §3）。

---

## 2. 评价逻辑的修正（Step 2）

上一轮的交叉表有两处**定义错误**，都会让结论失真：

| 处 | 旧实现 | 错误 |
|---|---|---|
| ① 判据 | `positive_correct = dist > min_pos[anchor]`（锚点相对） | 对正样本「比最近后继更远」被标为 correct，语义无意义；研究的应是逐对 `d(A,P) < d(A,N_k)` |
| ② 前缀共享 | `shared = prefix[anchor] == prefix[other]`（逐项布尔） | **正样本共享前缀（成功）与负样本共享前缀（失败）被并入同一格** |

修正为四类互斥分类（后两类是 **Tie**，不进错误计数），并给出 **pairwise**（主用，与逐对连续判据同构）与 **max-based**（更严的次用口径）两套 LCP 统计，各自 rates 与 AUC 自洽。LCP 只取前三个码本 token（第 4 个是 RecBole 碰撞 tie-breaker）。

### 修正后的结果（主用 valid 配对，52855×10）

| 模型 | 深度 | pos_adv | neg_adv | tie | 连续 AUC | LCP pairwise | LCP max | **连续正确→反转** | **连续正确→Tie** | 连续错误→量化正确 | ER |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Exp1a | L1 | 0.07407 | 0.04147 | 0.88446 | 0.776691 | 0.537303 | **0.516347** | 0.00787 | **0.79918** | 0.00353 | 16.39 |
| Exp1a | L1+L2 | 0.00341 | 0.00057 | 0.99603 | 0.521879 | 0.537303 | 0.516347 | 0.00032 | 0.97223 | 0.00021 | 58.39 |
| Exp1a | L1+L2+L3 | 0.00051 | 0.00006 | 0.99943 | 0.505661 | 0.537303 | 0.516347 | **0.0000** | **0.99507** | 0.0000 | 70.0 |
| TIGER | L1 | 0.07314 | **0.10879** | 0.81807 | 0.650415 | 0.536558 | **0.482528** | 0.01861 | 0.71462 | 0.00892 | 6.47 |
| TIGER | L1+L2 | 0.00891 | 0.00549 | 0.98560 | 0.541858 | 0.536558 | 0.482528 | 0.00191 | 0.94558 | 0.00248 | 16.0 |
| TIGER | L1+L2+L3 | 0.00195 | 0.00036 | 0.99769 | 0.537518 | 0.536558 | 0.482528 | 0.00016 | 0.98445 | 0.00013 | 50.0 |

**读数**：
1. **Q2 仍成立（量化不是瓶颈）**，但理由从「量化损失很小」换成了「**主要结果是 Tie（不可区分）而非错误**」：连续排对的比较中 79.9%/97.2%/99.5% 变为 Tie，真正的**反转只有 0.79%/0.03%/0.00%**。
2. 在**可区分的小子集**上前缀**偏向正样本**：Exp1a 的 max-based LCP AUC **0.516347**（与 main 独立复算逐位一致），而 TIGER 为 **0.482528** —— 行为监督**翻转了前缀排序的符号**，与它在 L1 上「无监督时 neg_adv 0.10879 > pos_adv 0.07314、被监督后反转」一致。
3. **LCP pairwise AUC 两模型几乎相同**（0.537303 vs 0.536558）：「整体前缀排序」不区分两者；区分度在 **L1 共享不对称性**（即 ER 16.39 vs 6.47）。
4. **四类失败严格分开**：连续排序错误（Exp1a L1 0.22331 / TIGER 0.34958）、量化反转、Tie/inconclusive、码字碰撞（0.036361 vs 0.177248）互不合并。
5. **重建视角**（独立第三视角）：Exp1a 0.773169、TIGER 0.656667 —— 该列按构造与深度无关（每商品只有一个累积重构向量），已标注，不作为逐层证据。
6. 次用（训练派生）配对集**显式标注 BLOCKED**（SIGKILL exit 137、峰值 RSS 128 GiB），未交付半成品。

---

## 3. 监督位置的单变量实验（Step 3）

唯一变量＝行为损失作用层（`ACTIVE_LEVELS` 门控），架构/数据/初始化/码本容量/训练配置全冻结；全层路径与改动前**逐位一致**（由 `scripts/test_level_gate.py` 内置的原实现副本钉死）。

### 3.1 逐层连续 AUC 与 SID（五臂 × 三层，主用 valid 配对）

| 臂 | 监督层 | L1 AUC [95% CI] | L2 AUC | L3 AUC | L1 码字占用 | 原始 SID collision |
|---|---|---|---|---|---|---|
| TIGER | 无 | 0.650415 [0.6473,0.6535] | 0.541858 | 0.537518 | 247/256 | 0.177248 |
| **Exp1a** | L1+L2+L3（= L1 @1/3） | **0.776691** [0.7743,0.7799] | 0.521879 | 0.505661 | 241/256 | **0.036361** |
| **l1only** | 仅 L1（**全强度**） | **0.776859** [0.7744,0.7799] | 0.524357 | 0.505837 | **196/256** | 0.045308 |
| `l2only` | 仅 L2 | 0.650415 | 0.541858 | 0.537518 | 247/256 | 0.177248 |
| `l3only` | 仅 L3 | 0.650415 | 0.541858 | 0.537518 | 247/256 | 0.177248 |

**`l2only`/`l3only` 与无监督基准逐位相同**（构造性结果，不是三个独立证据）。**有信息量的对照是 `l1only` vs Exp1a。**

### 3.2 Stage3（冻结协议：150 epoch、seed 42、beam 20、topk [5,10]、n_eval 57439）

| 臂 | L1 监督强度 | **test R@10** | R@5 | NDCG@10 |
|---|---|---|---|---|
| TIGER | 无 | 0.053674 / 0.054736 / 0.055050（三次历史 run） | — | — |
| **Exp1a** | 1/3 | **0.063633 / 0.063737**（本轮重跑） | 0.042549 | 0.035110 |
| **`l1only`** | **全强度** | **0.062832** | 0.041871 | 0.034597 |

**行为监督本身值约 +0.009**（对无监督）。但**把 L1 监督从 1/3 提到全强度反而下降 0.00091**，约为实测 run-to-run 噪声（同一 Stage2 产物、同协议两次 Exp1a 之差 **1.045e-4**）的 **9 倍**，且与独立测得的 SID 退化方向一致（码字占用 241→196、collision 0.0364→0.0453，+25%）。

⇒ **被这个缺陷掩盖的真实变量是「监督有效强度」，而 Exp1a 因三层取均值恰好落在接近最优的强度点上；沿该方向继续加强会损害下游。**

`l2only`/`l3only` 未跑 Stage3：其 Stage2 产物与无监督基准逐字节相同，重跑只会复现基准（0.0537–0.0551）。

---

## 4. 假设判定

| # | 假设 | 判定 | 依据 |
|---|---|---|---|
| H1 | 行为信息在**残差形成**阶段丢失 | **否证** | L1 残差（含全部编码信息）区分力达 0.777 |
| H2 | 行为信息在**量化**阶段丢失 | **否证** | 反转仅 0.79%/0.03%/0.00%，主结果是 Tie；且可区分子集上前缀方向正确 |
| H3 | 行为监督**从未有效到达** L2/L3 | **确认（根因）** | 梯度恰为零 + checkpoint/SID 逐字节等同基准 |
| H4 | L2/L3 值得施加行为监督 | **否证** | 可证明的 no-op；要生效必须换作用点 |
| H5 | Exp1a 是三层监督 | **否证** | 实为 1/3 强度的 L1 监督（均值把 L1 梯度缩 1/3） |
| H6 | 更强的 L1 监督更好 | **否证** | 全强度使 R@10 降 0.00091（约 9× 噪声），SID 同步退化 |
| H7 | 「拥挤」源于有效容量不足 | **仍缺证据** | 本轮无法区分容量/监督/量化；**不得**归因于欧氏几何 |

---

## 5. 本轮暴露并修正的实现缺陷

| # | 缺陷 | 后果 | 修法 |
|---|---|---|---|
| 1 | 量化桥接把正样本共享（成功）与负样本共享（失败）并入同一格 | 「量化损失」混同成功与失败 ⇒ 上一轮 Q2 的**数字**不可按原样采信 | 改为四类分类 + Tie 单独计数；11 项回归测试 |
| 2 | 「正确」判据用锚点相对（`dist > min_pos`） | 对正样本语义无意义 | 改为逐对 `d(A,P) < d(A,N_k)`，并列单独计 |
| 3 | LCP 方向/口径错误 | 报出 LCP AUC 0.4627（反向），实际应为 0.516347（正向） | 定位为口径混用，拆成 pairwise / max 两套并各自自洽；由 main 独立复算逐位校验 |
| 4 | Stage3 `_ddp_sync/run_dir.txt` **截断写**（`open(...,"w")` 先清零再写） | 并发读到空串 ⇒ rank 把 shard 写到父目录 ⇒ rank0 空等 30 分钟安全上限。**同一症状第三次出现**（前两次表现为 shard 落到上一次 run 的目录） | 改为**原子写**（临时文件 + `os.replace`），读取端拒绝空值 |

另：本轮未改动任何既有 checkpoint 或历史结果；Exp1a 与 TIGER 的产物经 sha256 核对未变。

---

## 6. 局限

1. **单次 Stage3**：每臂只跑一次，`l1only` 与 Exp1a 的 0.00091 差值基于一次运行对一次运行；虽约为同条件噪声（1.0e-4）的 9 倍，仍未做多种子复现（那属于「为否证结果做复现」，按纪律不擅自扩大）。
2. **单数据集、单架构族**：仅 Amazon 2023 Instruments、仅该 TIGER 变体族。
3. **深层的 ER 小样本**：L1+L2+L3 的 ER 建立在 4（Exp1a）/ 21（TIGER）个负事件上，已标 UNSTABLE。
4. **次用配对集 BLOCKED**：修正后的桥接只交付主用集合。
5. **straight-through 是本实现的性质**：其他 RQ-VAE 实现（无 STE 或对 codebook 路径解冻）不会自动呈现同样的梯度隔离，因此「L2/L3 不可监督」是**本实现的结论**，而非 RQ-VAE 的普遍结论。

---

## 7. 下一步（不在本轮实施）

1. **不要在残差上加深层行为监督**——它不会生效。若要利用深层行为信息，必须让约束作用在**码本/分配**上（上一轮已测：换度量可改变 9–40% 的码字分配，说明该位置是有杠杆的）。
2. **把「监督有效强度」当作一个真实变量来调**，并与 SID 指标联合看：本轮发现全强度 L1 监督在 R@10 与 collision 上同时变差，说明存在一个中间最优点，而 Exp1a 当前的 1/3 只是**因为均值取三层而偶然**落在附近。把这个自由度显式化，比继续加深层监督更可能有收益。
3. 若要让 STE 后的深层残差获得梯度，需要修改量化器本身（例如让残差绕过 STE 或对 codebook 路径解冻）——**那是新机制，本轮不引入**。

---

## 8. 产物与复现

```
analysis/crowding_diagnostic/
  quant_bridge.py / pair_prefix.py      修正后的量化桥接（四类 + 两种 LCP 口径）
  test_quant_bridge.py                  11 项回归测试
  supervision_location.py               五臂逐层 AUC / 码本占用 / SID
  test_supervision_location.py          含「L2/L3 与基准逐位相同」的构造性断言
stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/
  train_rqvae.py                        ACTIVE_LEVELS 门控（默认 (0,1,2)，A 逐位不变）
  scripts/test_level_gate.py            6 项：全层逐位一致、层隔离、梯度、mask 不变
results/crowding_diagnostic/quant_bridge_v2/        修正后的主用结果
results/crowding_diagnostic/supervision_location/   五臂对照表
results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/{resample_l1only,resample_l2only,resample_l3only}/
results/stage3_T5Train/tiger_sample_resample_l1only/logs/Amazon_2023_Instruments/Oct-12-2026_23-59-44/
```

复现：
```bash
cd /home/wlia0047/ar57/wenyu/GeneRec
P2=/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9
P3=/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10
$P2 stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/scripts/test_level_gate.py
$P2 analysis/crowding_diagnostic/test_supervision_location.py
$P3 analysis/crowding_diagnostic/test_quant_bridge.py
$P3 analysis/crowding_diagnostic/run_slice_b.py
# Stage2 每个臂：改 EXPERIMENT_ARM 后
cd stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE && $P2 train_rqvae.py
# Stage3：改启动器 ARM 后
cd /home/wlia0047/ar57/wenyu/GeneRec && $P3 stage3_T5Train/scripts/run_stage3_directional_arms.py
```

报告路径：`results/latent_interest/root_cause_report.md`