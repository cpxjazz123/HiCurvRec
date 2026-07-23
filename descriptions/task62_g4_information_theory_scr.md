# Task #62 — G4: 分段量化的信息论/高分辨率理论 — SCR 的闭式预测器 (理论线)

> **任务目的**: 验证 G4-P1: Gaussian 总相关 TC_G 与实测 SCR (Side-Channel Ratio) 单调相关 (T5 应低 TC, MCKG 应高 TC), 把"分段编码损失"从经验观测升级为可从协方差矩阵直接算出的量
> **执行日期**: (待启动, 优先级 1/5)
> **状态**: 🟡 待启动

---

## 1. 背景

承接 Task #58-#61 Campaign (最高 R@5 = 0.0977, 8 任务完成). 在 2026-07-20 四战线 (可量化性矩阵 / 三级修复 / 架构公平性 / TIGER 闭环) 框架下, 战线一是经验矩阵, **G4 是它的理论上层建筑**, 不重复任何实验, 纯粹消费已完成的数据, 产出闭式公式和可证伪预测.

**前置结论**:
- SCR (Side-Channel Ratio, 分段编码与联合编码的失真比) 是诊断 SID 跨 embedding 源难易度的核心标量
- 经验观测: 健康数据 (sentence-t5) SCR = 0.95x (分段 < 联合, 红利赢), 病态数据 (MCKG) SCR = 4.22x (分段 ≫ 联合, 病态税赢)
- **关键机会**: 把 0.95x vs 4.22x 从"两个观测"升级为"理论预测"

**假设 G4-H1 (可证伪)**: 跨 embedding 源, Gaussian 总相关 TC_G 与实测 SCR 单调相关. 数学:
$$
\mathrm{TC}_G = \frac{1}{2} \log \frac{\prod_{s=1}^{S} \det \Sigma_s}{\det \Sigma}
$$
($\Sigma$ 全维协方差, $\Sigma_s$ 各段块对角)

---

## 2. 实验设计

**变量**: TC_G 数值 (从 covariance matrix 算出)
**保持不变**:
- 所有 embedding (Task #58 S4 AE 64d, #59 flan-t5 2048d, #60 sentence-t5 768d, #61 Hybrid 2816d, + Task #87 MCKG 768d baseline)
- seed=42, K=256, num_hierarchies=3
- 已记录的 SCR 实测值

**计算工具 (纯 CPU, 无 GPU)**:
```bash
# D0: 算 TC_G + Zador 指数 (半天)
python3 scripts/task62_tc_g_zador.py \
    --embeddings logs/task58_s1/.../merged_predictions_tensor.pt \
                 logs/task59_s1/.../merged_predictions_tensor.pt \
                 logs/task60_s1/.../merged_predictions_tensor.pt \
                 logs/task61_s1/merged_predictions_2816d.pt \
                 logs/task87_s1/.../merged_predictions_tensor.pt \
    --out_csv verdicts/task62_tc_zador.csv

# P1: 全源矩阵的 TC/Zador 预测 vs 实测 SCR 拟合 (1-2 天)
# 输入: D0 CSV + 已记录 SCR 实测 (从战线一矩阵复制)
# 脚本: scripts/task62_scr_predict.py
# 输出: verdicts/task62_scr_predict_eval.json
```

---

## 3. 决策触发 (vs 提案 D0 GO 条件)

| TC_G 与 SCR 关系 | 决策 |
|------------------|------|
| TC_G 方向与 SCR 完全一致 (T5 TC_G ≈ 0, MCKG TC_G ≫ 0, 且单调) | ✅ GO → P1 全矩阵 + P2 理论 note |
| TC_G 方向部分一致 (一个例外) | ⚠️ PARTIAL → P1 残差分解 (SCR = f(TC_G) + tail term) |
| TC_G 方向不一致 (TC 低但 SCR 高) | ❌ NO-GO → 病不在二阶统计 (协方差), 在高阶/重尾效应 → 切换到 4.2 的 Zador 因子解释残差, 仍产出机制分解论文 |
| TC_G 与 SCR 完全无相关 | ❌ FALSIFIED → 关闭 G4 方向, 资源转 G1/G2 |

---

## 4. 预算

| 阶段 | 估算时间 | 备注 |
|------|---------|------|
| P0 文献核查 (1 周) | 半天 | 检索 `distributed quantization rate loss`, `Zador exponent learned embeddings` |
| **D0** (半天, GO 条件: TC_G 方向与 SCR 一致) | **0.5 天** | 纯 numpy.linalg, 无 GPU |
| P1 全矩阵拟合 | 1-2 天 | 仍纯 CPU, 可能引入 cvxpy 求极值 |
| P2 理论 note 成文 | 1 天 | proposition + 经验验证, 作为战线一论文的理论章节或独立短文 |
| **总计** | **3-4 天** | **零 GPU 时间** (纯消费数据) |

---

## 5. 风险与缓解

**风险 1**: 预测失败 (TC_G 低但 SCR 仍高) → 不视为失败, 而是产出"二阶统计 + 高阶/重尾残差分解"更深机制论文 (提案 4.3 已预分支)
**风险 2**: 行列式数值不稳定 (高维 embedding 协方差矩阵可能奇异) → 用 SVD + 截断小奇异值, 或加微小 jitter ($\Sigma + \epsilon I$)
**风险 3**: MCKG embedding 数据可访问性 → Task #87 S4 AE 64d + MCKG 数据路径需先确认

---

## 6. 完成度跟踪

- [ ] P0 文献核查 (检索词 + 邻近工作)
- [ ] D0 写 task62_tc_g_zador.py 脚本 + 跑 TC_G + Zador
- [ ] D0 GO/NO-GO 判定 (vs SCR 0.95x vs 4.22x)
- [ ] P1 全矩阵拟合 (如 D0 GO)
- [ ] P2 理论 note + proposition
- [ ] 写 verdict (含 result: 行)
- [ ] 更新 loop.md §16 (R8 归档)

---

## 7. 重叠审计 (vs 已杀/幸存清单 + 四战线)

- ✅ 与已杀清单零重叠 (whitening/MMQ/QINCo/ICA/SAE/collaborative injection/STACodec/GSRQ 均无信息论成分)
- ✅ 与四战线零重叠 (战线一经验矩阵 vs G4 理论上层建筑; 消费数据不重复实验)
- ✅ 与幸存 idea "Wyner-Ziv conditional rate allocation" 同属信息论家族, 不同题 (G4 诊断 vs WZ 设计方法), 可同章引用