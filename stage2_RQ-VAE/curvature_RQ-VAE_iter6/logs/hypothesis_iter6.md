# iter6 Hypothesis Document (Agent C: hypothesis-designer)

**任务**: 把 Agent B 唯一推荐 P1 (Sinkhorn linear ε-anneal + Lipschitz 检测) 翻译为可证伪假设。
**基线对照**: iter11 (sk_L0=0.50) test_R@10=0.0602 (+1.86% vs TIGER 0.0591), 0.065 hard target (差 ~8%)。
**上一轮 (iter5) dominant bottleneck**: attention temperature_scale 经 softplus + cyclic_factor.detach() + +0.25 bias 三联把 cyclic c(t) 联动淹没, attention logits 量级远小于 distances 量级。
**iter5 forbidden directions** (iter6 假设严禁触犯):
1. 再加 attention / codebook attention
2. softplus 长漂路径
3. cyclic_factor.detach() 截断反向
4. +0.25 bias 隐藏 cyclic 振幅
5. 任何隐藏 cyclic c(t) 振幅的常数偏置

---

## P1 机制摘要 (来自 Agent B direction_decision_iter6.md)

Sinkhorn assignment 层的 ε schedule 从 iter5 失败的指数衰减 + 三联 bias 切换为线性 stability law + 完整 sin 振幅:

```
ε(t) = ε_min + (ε_max - ε_min) · |sin(π t / T)|
```

- cyclic c(t) 与 ε(t) 通过显式乘法耦合可微, 振幅完整 1.0 峰谷差
- 无 detach / 无 softplus / 无常数 bias
- 当 c(t) 高 → ε 也高 → Sinkhorn 软分配保留多样性
- 当 c(t) 低 → ε 也低 → 软分配收紧利用低曲率几何紧致性
- 严禁 forbidden directions 全部 5 项

---

## 直接效应 (Direct Effects) — 可证伪硬验证点

### DE-1: ε(t) 与 cyclic c(t) 的线性相位耦合

**陈述**: cyclic c(t) 与 ε(t) 在 step 5000 的 Pearson 相关系数 |ρ| ≥ 0.95, 验证 ε(t) = ε_min + (ε_max-ε_min)·|sin(πt/T)| 公式正确执行, 没有被任何 detach / softplus / bias 截断。

**检测位置**: `[iter6][sinkhorn]` 日志每 1000 步打印一行 `(step, c, ε_actual)`, step 5000 同时打印 Pearson ρ(c, ε_actual)。

**判据**:
- PASS: |ρ(c, ε_actual)| ≥ 0.95 且 ε_actual 落在 [ε_min, ε_max] 区间内
- FAIL: |ρ| < 0.95 或 ε_actual 落在区间外或为常数

**理论依据**: c(t) = c_min + (c_max - c_min)·|sin(πt/T)| 与 ε(t) 同公式同相位, 两者必线性相位耦合, ρ 应趋近 1.0; iter5 失败因 softplus+detach+bias 三联使 ρ ≈ 0。

### DE-2: Sinkhorn assignment Q 在 cyclic c(t) 影响下周期性变化

**陈述**: Sinkhorn assignment 矩阵 Q = softmax(log P / ε) 在 step 1000/2000/3000/4000/5000 打印 min/median/mean/row 四分位, 数值随 step 周期性震荡 (峰谷差 ≥ 20% 相对值), 且 max(Q)/mean(Q) 比值与 c(t) 单调反向 (高 c → 更平 → 比值接近 |C|, 即接近均匀)。

**检测位置**: `[iter6][sinkhorn]` 日志每 1000 步追加打印 Q 的统计量, 同步记录 c(t) 值。

**判据**:
- PASS: 5 个采样点 Q 的 mean 在 [Q_min, Q_max] 区间内震荡, 且 max/mean 比值与 c(t) 序列的 Pearson ρ ≤ -0.7
- FAIL: Q 数值无震荡 (std/mean < 5%) 或 max/mean 比值与 c(t) 不单调反向

**理论依据**: ε 高 → softmax 更接近均匀 → max(Q) 趋近 mean(Q), 比值下降; ε 低 → softmax 更尖锐 → max(Q) 远大于 mean(Q), 比值上升; c(t) 高 → ε 高 → 比值低 (反向单调)。

### DE-3: 训练稳定性 — 3 token unique 不低于 iter5 同期

**陈述**: Step 5000 时 SID 文件 3-token unique count ≥ iter5 同期 baseline (避免 L0 collapse 失控)。

**检测位置**: `[iter6][sid]` 日志在 step 5000 打印 3-token unique 数量与 iter5 同期对照。

**判据**:
- PASS: 3-token unique (iter6, step 5000) ≥ 3-token unique (iter5, step 5000)
- FAIL: 3-token unique 低于 iter5 同期

**理论依据**: ε(t) 与 c(t) 同相位震荡, c 峰时刻 ε 峰, 分配保留多样性 → 不会因 ε 收紧导致 L0 collapse 失控; iter11 强 L0 collapse 是几何 fingerprint, 不是禁忌。

---

## Proxy 假设 — 描述性辅助信号

### PH-1: H(L1|L0) 保持或略升

**陈述**: Step 5000 时 SID 的 H(L1|L0) (L1 给定 L0 时的条件熵, 单位 bits) 与 iter5 同期 baseline 相比保持或略升 (容差 ≥ -0.05 bits)。

**检测位置**: `[iter6][proxy]` 日志在 step 5000 打印 H(L1|L0), 与 iter5 同期对照。

**判据**:
- PASS: H(L1|L0)(iter6, step 5000) ≥ H(L1|L0)(iter5, step 5000) - 0.05
- FAIL: H(L1|L0) 下降超过 0.05 bits

**理论依据**: ε 高 → 软分配保留多样性 → L1 在 c 峰时刻分配更广 → L1 给定 L0 条件熵上升或保持; iter11 fingerprint H(L1|L0)=4.39 (强 diversity) 是参考基准。

### PH-2: collision rate 不增加

**陈述**: Step 5000 时 SID 文件 collision rate (1 - unique_pairs / total_pairs) 与 iter5 同期相比不增加 (容差 ≤ +0.5%)。

**检测位置**: `[iter6][proxy]` 日志在 step 5000 打印 collision rate, 与 iter5 同期对照。

**判据**:
- PASS: collision_rate(iter6) ≤ collision_rate(iter5) + 0.5%
- FAIL: collision_rate 增加超过 0.5%

**理论依据**: ε 振幅完整保留多样性, 不会因分配过紧导致 collision 暴涨; v342 R36p Sinkhorn-OT 失败是课程化太激进, 本候选 |sin| 完整振幅避免短路径 util 崩塌。

---

## 不可证伪词扫描

本文件不含 "可能提升" / "也许" / "视情况而定" / "或许" / "大概" / "似乎" 等不可证伪词。

---

## 完成声明

- 假设落盘: `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter6/logs/hypothesis_iter6.md`
- 直接效应数量: **3** (DE-1 / DE-2 / DE-3)
- Proxy 假设数量: **2** (PH-1 / PH-2)
- 严禁方向: iter5 全部 5 项 forbidden directions
- 完成时间: 2026-09-21
- 设计者: Agent C (hypothesis-designer)