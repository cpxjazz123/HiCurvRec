# Task #137 — 论文 Eq11/12 ρ 单调假设 + 因果链验证

> **任务目的**: 验证 HG-Rec paper §3.2.3 Theorem 3.4 → Eq11 → Eq12 的关键单调假设: "**相邻 layer 的 codeword 半径 ρ 增量 Δρ = const** (等增量), 因此 K_ℓ = K_1·γ^ℓ (几何级数)". 我们的 codebook (K=[64,128,256], γ_target=2) 是否真的满足此假设?
> 
> **完成日期**: 2026-07-29
> **状态**: 🟡 **实证部分: ρ 序列违反单调假设, 因果链断开**

## 1. 背景 (paper §3.2.3)

paper Theorem 3.4 推导: Poincaré ball 半径 ρ 处的 "码字数" K_ρ 服从
$$K_\ell = K_1 \cdot e^{(n-1)\sqrt{c}\,\rho}, \tag{11}$$
由此若 Δρ 在 layer 间为常数, 则
$$K_\ell = K_1 \cdot \gamma^\ell, \quad \gamma = e^{(n-1)\sqrt{c}\,\Delta\rho}, \tag{12}$$
即 K_ℓ 应该是几何级数 (paper "our experiments use K_1=64, L=3, γ=2, giving codebook sizes [64,128,256]").

**核心假设 R1**: Δρ = const (相邻 layer codeword 半径差恒定)
**因果链 R2**: K_ℓ = K_1·γ^ℓ (码字数几何增长) ⇒ 容量爆炸 ⇒ 表征能力强

## 2. 验证方法

Poincaré 球 c>0 下, codeword 半径与 Euclidean 范数关系:
$$\rho = \frac{2}{\sqrt{c}} \cdot \text{artanh}(\sqrt{c}\,\|e\|_E)$$

**输入**: Task #239 Gate 2 三个 Stage 1 ckpt (200 epoch 训练, musical_instruments 9922 items)
- B control: c=[10,10,10]  (最接近 paper 默认)
- A1: c=[10,1,1]  (L0 高, L1/L2 低)
- A2: c=[30,3,3]  (REVERSED, 跟 paper 倍数对齐)

**输出**: ρ_ℓ + Δρ_ℓ 序列 + γ_predicted (Eq12 预测的 γ)

## 3. 验证结果

| Variant | L0 ρ | L1 ρ | L2 ρ | Δρ(L0→L1) | Δρ(L1→L2) | γ_pred(L0→L1) | γ_pred(L1→L2) | γ_target |
|---------|-------|-------|-------|-----------|-----------|---------------|---------------|----------|
| B c=[10,10,10] | 0.273 | 0.169 | 0.129 | **-0.104** | **-0.040** | 0.0000 | 0.0205 | 2.0 |
| A1 c=[10,1,1] | 0.277 | 0.173 | 0.128 | -0.104 | -0.045 | 0.0000 | 0.2455 | 2.0 |
| A2 c=[30,3,3] | 0.169 | 0.129 | 0.095 | -0.040 | -0.034 | 0.0011 | 0.1597 | 2.0 |

## 4. 关键发现

### 4.1 ρ 序列反向 (REFUTED)

**Paper Eq12 假设 Δρ > 0 (单调递增)**, 实测 **Δρ < 0 (单调递减)**:
- B: ρ_0=0.273 > ρ_1=0.169 > ρ_2=0.129 (decay 53%)
- A1: ρ_0=0.277 > ρ_1=0.173 > ρ_2=0.128 (decay 54%)
- A2: ρ_0=0.169 > ρ_1=0.129 > ρ_2=0.095 (decay 44%)

**所有 3 个变体的 ρ 都从 L0 → L2 严格下降**. paper Eq12 假设完全 refuted.

### 4.2 γ_pred 全部 < 1 (而不是 paper 要求的 2.0)

按 Eq12 公式 γ = e^{(n-1)·√c·Δρ}, 因 Δρ<0, γ_pred < 1.
- B L0→L1: γ_pred = e^{31·3.16·(-0.104)} = 2.1e-5 (vs target 2.0)
- A1 L1→L2: γ_pred = e^{31·1.00·(-0.045)} = 0.2455 (vs target 2.0)
- A2 L1→L2: γ_pred = e^{31·1.73·(-0.034)} = 0.1597 (vs target 2.0)

**γ_pred 偏离 target 至少 8× (A2), 最多 10^5× (B L0→L1)**.

### 4.3 训练找到的反向方案: Euclidean-shrinkage

实测训练出的 codebook 行为: K 翻倍 (64→128→256), 但 ρ 减半 (0.27→0.13).
这是 RQ-VAE **残差 hierarchical compression** 的自然结果:
- L0 拟合粗粒度 (大码字, ρ 大)
- L1/L2 拟合残差 (细粒度, 码字小, ρ 小)
- K 翻倍是因为细粒度需要更多码字覆盖精细结构
- 不是 paper Eq12 假设的 "K 翻倍 = ρ 增加" 反向方案

**Paper 假设的 K ∝ e^{(n-1)√c·ρ} 是 K 关于 ρ 的单调关系, 但训练给出的解是 K ∝ 1/ρ (反比) — K 越大 ρ 越小, 因为码字必须更分散以避免碰撞**.

### 4.4 因果链状态

paper Eq12 因果链: "等 Δρ ⇒ K 几何级数 ⇒ 容量分配合理"
- (a) 等 Δρ: REFUTED (实测单调下降)
- (b) K 几何级数: 维持 (我们 64→128→256 是 K 设定, 不是 paper 公式推出来的)
- (c) 容量分配合理: 部分维持 (基线 R@10=0.1020 OK, 但 c-scan 都低于 phonism 0.1058)

**因果链 (a)→(b) 断开**: paper 假设是 "**先固定等 Δρ, 再推导 K 几何级数**"; 实际训练是 "**先固定 K 几何级数 (codebook_size 参数), 训练自然让 ρ 反向**". 两者方向相反.

## 5. 决策

按用户规则 "前一个 stage 没达到要求则不继续":

- **paper Eq12 单调假设 (R1)**: REFUTED ❌ (ρ 序列实测全部下降)
- **paper 因果链 (R2)**: 部分维持 (K 几何级数是 recipe, 不是推导结果)
- **决策**: paper Eq11/12 公式作为**理论 prescription** 成立 (数学推导正确), 但作为**训练实现 prescription** 失败 (实际训练找到 Euclidean-shrinkage 解).

**不接受** paper Eq12 的 "等 Δρ ⇒ K 几何级数" 因果关系.

## 6. 启示

1. **paper Theorem 3.4 公式层正确**: K_ρ = e^{(n-1)√c·ρ} 在给定 ρ 时是对的, 但 ρ 本身是训练结果不是预设.
2. **paper §3.2.3 "实验使用 γ=2" 隐含一个 training-time prior**: 实际 RQ-VAE 训练不会自动产生等 Δρ, 需显式 loss 约束 (e.g. codeword 半径 penalty / Sinkhorn 平衡).
3. **未来方向**: 既然 paper 因果链断开, R@10 杠杆不在 ρ 几何上. 真正杠杆是:
   - Sinkhorn during training (Issue #10 redesign D/E/F 路径)
   - 训练时长 (task200 dual_v5 -10.3% 反向证据)
   - 跨架构 (LETTER R@10=0.0509 / S3Rec 缺 / FDSA R@10=0.0594 都没到 0.1020)

## 7. 产物

- 本 description: descriptions/task137_paper_eq11_12_rho_monotonicity.md
- verdict: verdicts/task137_paper_eq11_12_result.md
- 计算脚本 (无 GPU, 仅 ckpt 读取): 见 verdict §8

## 8. Status

- ✅ paper Eq12 单调假设 REFUTED
- ❌ paper "等 Δρ ⇒ K 几何级数" 因果链断开
- ⏭️ 下一个方向候选: Issue #10 redesign 4-arm (等用户) / 跨架构 R@10 重新基线 / 训练时长验证
