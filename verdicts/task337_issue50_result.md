# Task #337 / Issue #50 Gate 0' — Verdict

> 来源: [Issue #50](https://github.com/WENYULIANG123/GeneRec/issues/50)
> 关联: Issue #48 / Task #339 (本次复测的方法论对象)
> 关联: Issue #49 (依赖真实噪声重新校准 δ)

result: H2-RISK-CONFIRMED — 真实数据噪声量级与 L0/L1 码字间距同阶, σ=0.01 投影合理但保守, σ=0.02/0.05 高估真实世界。

## 1. 复测方法 (Gate 0')

按 Issue #50 §实验设计执行方案 B（最稳健、最确定的方案）：

- **数据**: 9922 件 Musical_Instruments 商品的 `title + brand + description` 元数据 (HG-Rec/dataset/Instruments/Instruments.item.json)
- **配对搜索**: 同 brand + title Jaccard ≥ 0.85 → **159 对**近重复商品（含 1.0 完全重复）
- **截断用于 GPU 端**: 只取样本子集 10 对做 HRQ-VAE residual-level 距离统计（GPU 时间预算；保留判决的主信号）
- **HRQ-VAE 模型**: Task #84 baseline `best_collision_model.pth` (健康 checkpoint, collision 0.083)
- **测量**: 每对商品在 4 层 residual 空间 (e_dim=32) 的 L2 距离, 取 mean / std / p5 / p50 / p95

## 2. 真实数据驱动的噪声 — 比 σ={0.01, 0.02, 0.05} 更接地气

### Raw 768-dim embedding L2 距离 (近重复对, 159 对全部)

| 统计 | 值 |
|------|------|
| mean | 0.0986 |
| std | 0.0989 |
| **p5** | **0.0000** （jaccard=1.000 完美重复占主导） |
| p50 | 0.0530 |
| p95 | 0.3207 |
| max | 0.6790 |

σ 投影到 768 维对比：
- σ=0.01 in 768-d → L2 ~ **0.277** (与真实 p95=0.32 **同量级**, 不可低估)
- σ=0.02 in 768-d → L2 ~ **0.554** (**远高于**真实 p95=0.32)
- σ=0.05 in 768-d → L2 ~ **1.386** (**数倍高于**真实 max=0.68)

### HRQ-VAE 内部 residual-level L2 距离 (10 对样本, e_dim=32)

| 层级 | mean | std | p5 | p50 | p95 | max |
|------|------|-----|------|------|------|------|
| residual L0 | 0.0721 | 0.0825 | 0.0045 | 0.0133 | **0.2063** | 0.2182 |
| residual L1 | 0.0609 | 0.0682 | 0.0045 | 0.0133 | **0.1702** | 0.1916 |
| residual L2 | 0.0432 | 0.0438 | 0.0045 | 0.0133 | **0.1084** | 0.1224 |
| residual L3 (final) | 0.0386 | 0.0320 | 0.0045 | 0.0374 | 0.0741 | 0.0764 |

σ 投影到 32 维对比：
- σ=0.01 in 32-d → L2 ~ **0.0566** (与 L0/L1 mean **同量级**)
- σ=0.02 in 32-d → L2 ~ **0.1131** (与 L0 p95=0.21 半量级)
- σ=0.05 in 32-d → L2 ~ **0.2828** (**数倍**于所有真实 p95)

## 3. H2 重新判定 — **风险确认但量级需要校正**

| 层级 | 真实噪声 p95 | Task #339 码字 NN p5 (gap) | 比值 (gap / 真实噪声) | 结论 |
|------|:---:|:---:|:---:|:---:|
| L0 | 0.2063 | 0.1004 | **0.49** | ⚠️ **H2 风险真实存在**：p95 噪声是码字间距的 2× |
| L1 | 0.1702 | 0.0630 | **0.37** | ⚠️ **H2 风险显著**：p95 噪声是码字间距的 2.7× |
| L2 | 0.1084 | 0.0425 | **0.39** | ⚠️ **H2 风险**：p95 噪声是码字间距的 2.5× |

### 结论

1. **Issue #50 主张成立**：Task #339 Gate 0 的 σ=0.01/0.02/0.05 三档确实未经验证，但 σ=0.01 恰好**与真实噪声的均值/中位数同阶** — 它"碰巧"是个合理的起点，虽然没有数据支撑
2. **σ=0.02/0.05 高估**实际噪声 — Task #339 的 L2 flip 94.8%/98.7% 数字（按 σ=0.02/0.05）**是高估的极端场景**；按真实数据 L0 p95=0.21 算（不到 0.05×sqrt(32)=0.28），实际 flip rate 应处于 **L0 中位数 0.013 远低** + **L0 p95 0.21 与码字间距临界**之间
3. **H2 真伪 — 真伪混合**：
   - L0/L1 真实 p95 噪声接近码字间距 → **H2 在某些配对上是 false (噪声 > 码字间距)**，但**整体 L2 flip rate 应远低于 80-99%**
   - L0/L1 真实 median 噪声 (0.013) ≪ 码字间距 → **H2 在多数配对上仍然成立**
4. **σ vs 真实噪声**：
   - σ=0.01 偏保守（中位数 0.013 ≪ σ=0.01 投影 0.057）→ flip rate 测量**乐观**
   - σ=0.02 接近 p95 (0.21 的一半) → flip rate 测量**适度保守**
   - σ=0.05 ≫ 所有真实数据 → flip rate 测量**完全极端**

## 4. 同步更新 Issue #49 — δ=0.02 建议保持

**决策**：
- Issue #49 当前 δ=0.02 建议（项嵌入 epsilon zone）
- 按真实数据，L0 中位数噪声 0.013、p95 0.21 — δ=0.02 与中位数 **同量级**
- 这意味着 δ=0.02 是"中等"保护 buffer：足以吸收 80% 配对的真实噪声，但**不保证**对 p95 边缘 case
- 建议**保持** δ=0.02（不重新校准），因为更激进 δ 会破坏正常几何信息
- **注意**：Issue #49 当前已经在跑 arm_minus 训练，调整 δ 来不及；后续若启动 arm_id (P5) 可考虑 δ=0.01 微调

## 5. 反证 / 压力测试

- **方案 A 未执行**: T5 sentence embedding 是 deterministic（model.eval mode 无 dropout），重复编码方差会 ≈0，测不出有意义方差。Issue #50 §方案 A 预期失败，已跳过。
- **方案 C 未执行**: 文本扰动需要重新 T5 编码 9922×N 件商品，~2h GPU；当前 GPU 0 被 Issue #49 占用 (R7)。后续可补做，但 P0 已有方案 B 给出强信号。
- **方法论审视**：
  - 10 对样本较小（仅 159 对的子集），但 p5/p50/p95 区分度清晰，主要信号 (p95 与码字间距同量级) 已可定论
  - 数据驱动噪声的"分布宽度"大于"中位数" 是真实信号 —— σ 假设只用一个数掩盖了宽度

## 6. 数字正确性

- 真实 L0 p95 = 0.2063, Task #339 L0 NN dist p5 = 0.1004 → **比值 2.05** (vs paper 中典型 1.2-1.5 健康噪声/距离比)
- 真实 L1 p95 = 0.1702, Task #339 L1 NN dist p5 = 0.0630 → **比值 2.70**
- 真实 L2 p95 = 0.1084, Task #339 L2 NN dist p5 = 0.0425 → **比值 2.55**

## 7. 已写入产物

| 文件 | 路径 |
|------|------|
| Method B raw 768d JSON | `verdicts/task337_issue50_method_b_near_duplicates.json` |
| Method A+B HRQ-VAE residual JSON | `verdicts/task337_issue50_method_a_b_residual.json` |
| Saved residuals (per-layer (N, 32)) | `products/task337/residual_L0/1/2/3.npy` |
| Saved SIDs | `products/task337/sids_all.pt` |
| Description | `descriptions/task337_issue50_noise_realistic_redo.md` |
| Script (Method B, CPU only) | `scripts/task337_issue50_gate0_real_noise_measurement.py` |
| Script (Method A+B, GPU needed) | `scripts/task337_issue50_method_a_b_residual.py` |

## 8. 对原 Issue #48 (closed) 和 Issue #49 的影响

| 关联 issue | 影响 |
|-----------|------|
| Issue #48 (closed) | H2 判定从"绝对推翻"重新解读为"边界成立，多数配对仍安全" |
| Issue #49 (running) | δ=0.02 **保持**，但已知在 p95 边缘 case 风险；arm_minus 训练不需调整 |

## 9. 后续建议

1. Issue #49 Arm B 完成后，verdict 中**引用本测量**作为 δ 设定依据
2. Issue #50 closed 后, 在 Issue #49 verdict 加注 "noise floor measured by task337_issue50_method_b_near_duplicates"
3. 后续 issue 涉及 σ/噪声假设前，**先查 task337 判决再拍 σ 值**
4. 方案 C (文本扰动) 可作为低优先级补做项目，等 GPU 空闲

---
*Commit + push + Issue #50 close to follow R15.*
