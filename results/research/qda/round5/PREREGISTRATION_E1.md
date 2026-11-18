# 预登记:E1(阈值版漂移约束)——写于任何 E1 结果之前

## 唯一改动

`D2` 的漂移惩罚换成阈值版,其余(λ、快照缓冲、argmin 量化、学习率、预算)**逐字不变**:

$$L_{\text{drift},\tau}=\frac1N\sum_i \left[\max\!\left(0,\ \frac{\lVert z_i^t-\mathrm{sg}(z_i^{t-1})\rVert}{\lVert \mathrm{sg}(z_i^{t-1})\rVert}-\tau\right)\right]^2,\qquad \tau=0.01,\ \lambda=0.04$$

- `τ = 0.01` 来自**已测量**的码本移动率(0.1–0.6 %/epoch,取 1 % 即最快观测值的约两倍)。它**不是**已证明的"可追踪速度上限"——这一点在报告里必须照写,事后也不许因为结果不好而改 τ。
- `λ` 沿用 D2 的 0.04(在 epoch 1 与重建项等量)。改形状不改权重,保证与 D2 的对比只差一个 `τ`。

## 通过条件(三条必须同时成立)

以 `A_fine`/`B0` 的 300-epoch 口径为准(B0:F0 原始配方,L1 26 码、encoder 0.6594、L3 0.6592):

| 编号 | 判据 | 阈值 | 依据 |
|---|---|---|---|
| **C1** | epoch 300 的 L1 活跃码 | **≥ 128**(基线的 4.9 倍) | 若阈值版真的只抑制危险漂移,占用应接近 D2 的 198 而非基线的 26 |
| **C2** | encoder 行为 AUC | **≥ 0.6575** | "恢复接近 0.659",且不劣于 D1 的 0.6579(留 0.0004 余量) |
| **C3** | L3 行为 AUC | **≥ 0.6587** | F0 的 0.6592 减去 0.0005 的屏幕间重复差(本系列 300-epoch 同配方重复实测 0.0005 量级) |

**判定规则**

| 结果 | 结论 |
|---|---|
| C1∧C2∧C3 | 阈值版**成功分离**了"导致坍缩的漂移"与"学习所需的漂移";下一步才允许把该机制写进候选 |
| C1∧¬C2 | 阈值版挡住了坍缩但仍在抑制学习 ⇒ 需要更小的作用面(例如只作用于 L1 分配边界),不是调 τ |
| ¬C1 | 阈值版没有挡住坍缩 ⇒ "1 %/epoch"这个上限的假设不成立,应改用**实测**的每商品可追踪位移 |
| C2∧C3∧¬C1 | 学习没被抑制但坍缩照旧 ⇒ 说明坍缩并非由 >1 %/epoch 的漂移单独驱动,与 D2 的因果结论冲突,必须重新检查 D2 |

**不作的事**:不调 τ、不调 λ、不换种子、不做多种子、不启动 Stage3、不改 Stage2 正式候选。

## 复现

```bash
<genrec_env_v2>/python3.9 results/research/qda/round5/build_d_arms.py   # 四个臂,D 三臂路径不变
cd results/research/qda/round5/trees/E1_hinge_tau && <genrec_env_v2>/python3.9 train_rqvae.py
<genrec_env_v2>/python3.9 results/research/qda/round2/analyze_round2.py \
    --runs results/research/qda/round5/runs --arms E1_hinge_tau D2_drift B0_replay --no-strata \
    --out "$PWD/results/research/qda/round5/report_e.json"
```
