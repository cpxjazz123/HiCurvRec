# 预登记:F1 / F2(码本跟随编码器的分布漂移)—— 写于任何 F 结果之前

## 机制与无参性

不再限制编码器。每一步 `optimizer.step()` 之后,测量**本步被选中的码字实际移动了多少**(逐步均值 `m_ℓ`),把它加到本步够不到的码字上:

$$m_\ell=\frac{1}{|\mathcal{A}_\ell|}\sum_{k\in\mathcal{A}_\ell}\left(c_k^{(t)}-c_k^{(t-1)}\right),\qquad
c_k \leftarrow c_k+m_\ell \quad (k\in\text{目标集合})$$

| 臂 | 目标集合 | 含义 |
|---|---|---|
| `F1_dead_shift` | 本步**未被选中**的码字 | 把分布漂移传给死码,活码照旧只走梯度 |
| `F2_rigid_shift` | **全部**码字 | 整张码本随分布做刚性平移 |

- **没有需要标定的常数**:`m` 是当步实测的位移,不含衰减、窗口、阈值。
- 量化规则、损失、学习率、预算全部与 F0 相同;仍然只用 argmin,因此 L1/L2 的训练/导出一致性保持 1.0000。
- 已知的对照:`B4_revival`(第二轮的"死码重放")同样恢复了 256 个码,但 L1 行为保真没有改善(+0.79 pp 缺口)。F 与 B4 的区别是**跟随**而非**重放**:码字被连续搬运,而不是被重新放到某个商品上。

## 唯一主判据(按本轮指示:L3 行为 AUC)

| 编号 | 判据 | 阈值 | 依据 |
|---|---|---|---|
| **P** | epoch 300 的 **L3 行为 AUC** | **≥ 0.6597** | F0 的 0.6592 加 0.0005 的屏幕间重复差 |

诊断量只记录、不作闸门:L1 活跃码、L1/L2/L3 各自 AUC、L1 量化缺口、L3 collision、训练/导出一致率。

## 判定规则

| 结果 | 结论 |
|---|---|
| 任一臂 P 通过 | 码本跟随是有效机制;下一轮取较好的一臂做 3000 epoch + 种子检验 |
| 两臂都未通过但 L3 AUC ≥ F0 | 中性;记录为"跟随不改变最终结构保真",不再扩展 |
| 两臂 L3 AUC < F0 | 负结果:让码本跟随分布不能替代"码字必须被数据选中"这一约束 |
| 活跃码大幅提高而 L3 不提高 | 与 B4 同型:容量不是瓶颈,关闭这条线 |

**不作的事**:不调 `m` 的系数、不加衰减、不改阈值、不换种子、不做多种子、不启动 Stage3、不改正式候选。

## 复现

```bash
<genrec_env_v2>/python3.9 results/research/qda/round6/build_f_arms.py
cd results/research/qda/round6/trees/F1_dead_shift && <genrec_env_v2>/python3.9 train_rqvae.py
<genrec_env_v2>/python3.9 results/research/qda/round2/analyze_round2.py \
    --runs results/research/qda/round6/runs --arms F1_dead_shift F2_rigid_shift --no-strata \
    --out "$PWD/results/research/qda/round6/report_f.json"
```
