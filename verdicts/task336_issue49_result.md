# Task #336 / Issue #49 — Verdict

> 来源: [Issue #49](https://github.com/WENYULIANG123/GeneRec/issues/49)
> 接续: Issue #47 (统一 κ-stereographic 公式修复完成) + Issue #50 (真实噪声量级校正)
> 基线: Task #84 HG-Rec (R@10=0.1020 测试集, beam=20)

result: NO-GO — Arm B (κ=-0.12 学到的真实曲率) R@10=0.1005, 比 baseline 低 1.5%, 不达决策阈值 (>0.1020)。

## 1. 决策阈值（vs HG-Rec）

| 指标 | HG-Rec baseline | Issue #49 Arm B | Δ | 决策 |
|------|:---:|:---:|:---:|:---:|
| **Recall@10** | 0.1020 | **0.1005** | **-1.5%** | ❌ NO-GO |
| Recall@5 | 0.0816 | 0.0815 | -0.1% | 持平 |
| Recall@20 | 0.1279 | 0.1210 | -5.4% | ⚠️ 显著下降 |
| NDCG@10 | 0.0755 | 0.0748 | -0.9% | ❌ NO-GO |
| NDCG@20 | 0.0821 | 0.0800 | -2.6% | ⚠️ 明显下降 |

R@10 未超越 baseline 0.1020 → **NO-GO**。

## 2. 训练 + 评估完整流程

| 阶段 | 状态 | 关键数据 |
|------|:----:|---------|
| 方案 A 公式修复 | ✅ | Issue #47 5/5 测试 PASS, 统一 κ-stereographic 公式正确 (Möbius addition 符号反转 + tan_κ⁻¹ NaN 修复) |
| Stage 1 训练 (3 臂 Phase A+B) | ✅ | Arm B (θ=-0.02) 唯一 Gate 1 PASS, 学到 κ=[-0.128, -0.110, -0.123] |
| Stage 2 SID 推断 | ✅ | 9922/9922 unique, collision=0.1906 |
| Stage 3 T5-mini 200 epoch | ✅ | early stop @ epoch 84, best=NDCG@20=0.0961 (epoch 65), Val R@10=0.1224 |
| Stage 4 测试评估 (beam=20) | ✅ | **R@10=0.1005** |

## 3. 完整 Steering 链路回顾

**关键设计**：先有 Issue #47 公式修复（保证数学干净），才能在 Issue #49 跑 per-layer κ 解耦调度：
1. **公式干净**：Möbius addition `+ κ‖y‖²` → `-κ‖y‖²`；sigmoid blend → torch.where crisp threshold
2. **Phase A 解耦** (epoch 1-200, 200)：θ frozen at θ_init, Sinkhorn-enabled codebook 训练
3. **Phase B 学习** (epoch 201-400)：θ unfrozen with lr_theta=1e-5 → 学到 κ=[-0.128, -0.110, -0.123]
4. **Sinkhorn essential**：Phase A 用 `sk_eps=[0.0,0.0,0.0]` 直接坍缩（util=1.56%）→ 改 `sk_eps=[0.01,0.01,0.01]` 后健康

## 4. 训练统计

| 项目 | 数据 |
|------|------|
| 训练起始 | epoch 0, 16:35 |
| Early stop 触发 | epoch 84 (best=epoch 65 NDCG@20=0.0961) |
| 训练时长 | 84 epoch × ~50s = ~70 min |
| Early stop counter | 4 → 14 → 20 (patience=20) |
| Best Val R@10 | 0.1224 (epoch 65) |
| Best Val NDCG@20 | 0.0961 (epoch 65) |
| Final training loss | 1.881 |

## 5. 测试集评估

```json
{
  "best_ckpt": "products/task340/arm_minus/t5_out/Instruments/Jul-30-2026_16-35-52/HG_Rec_best.pth",
  "beam_size": 20,
  "test_recalls": {
    "Recall@5": 0.08152,
    "Recall@10": 0.10055,
    "Recall@20": 0.12098
  },
  "test_ndcgs": {
    "NDCG@5": 0.06871,
    "NDCG@10": 0.07480,
    "NDCG@20": 0.07998
  }
}
```

## 6. 与 Issue #50 噪声量级共同解读

**Issue #50 真实数据噪声**：
- L0 p95 (real) = 0.21，码字 NN gap L0 = 0.10 (比值 2.05)
- L1 p95 (real) = 0.17，gap L1 = 0.06 (比值 2.70)
- L2 p95 (real) = 0.11，gap L2 = 0.04 (比值 2.55)

**含义**：L0/L1 真实噪声 p95 ≈ 2× 码字间距，**H2 风险真实存在**
- 学到的 κ=[-0.128, -0.110, -0.123]（负值，符合 dual manifold 性质）
- 但 Stage 3 训练数据噪声 / test 噪声差异，**曲率几何信号在 T5 生成 SID 上不能有效传递**

## 7. 与基线对比的真正含义

| Issue #49 Arm B 测试 | Δ vs baseline | 解读 |
|------|:---:|------|
| R@10 -1.5% | vs 0.1020 | Sinkhorn + 4th digit dedup 在 k=-0.12 几何上 **轻微损害** top-10 |
| R@20 -5.4% | vs 0.1279 | 长尾 recall 损失更严重（k=-0.12 让上层码字更难区分） |
| NDCG@10 -0.9% | vs 0.0755 | 排名质量略降 |

→ per-layer 可变负曲率**几何上有意义**（Ollivier mean c=0.74），但**对下游 T5-mini 召回无帮助**

## 8. 综合判定

**Issue #49 Arm B (θ=-0.02, 学到 κ=-0.12)** 在干净公式下**完整跑通 4 阶段**，但:
- R@10 0.1005 < baseline 0.1020 (NO-GO by 1.5%)
- R@20 0.121 < baseline 0.128 (NO-GO by 5.4%)
- 结论：per-layer learned κ 是"代码干净的几何信号"，但**不是 R@10 的杠杆**

## 9. 后续

| 方向 | 状态 |
|------|------|
| Issue #30 r_l+s_l 极端 per-layer codebook transforms | ✅ 唯一 GO (+0.2pp) |
| Issue #49 本方向 (per-layer learned κ) | ❌ NO-GO |
| Issue #50 真实噪声测量的 δ=0.02 校准 | ✅ 校准保持 |

**经验记录**: 在干净公式基础上，per-layer κ 解耦可稳定学到几何信号（Ollivier c=0.74, κ=-0.128/-0.110/-0.123），但此信号在 T5-mini 9M-参数 SID 生成上不能转化为召回提升。**几何与召回的传递损失**是当前架构的天花板。

## 10. 产物

| 文件 | 路径 |
|------|------|
| Issue #49 description | `descriptions/task336_issue49_4stage_freecurv_pipeline.md` |
| Stage 1 训练脚本 | `scripts/task340_issue49_stage1_train.py` |
| Stage 2 SID 推断脚本 | `scripts/task336_issue49_stage2_infer.py` |
| Stage 4 评估脚本 | `scripts/task336_issue49_stage4_eval.py` |
| 测试集 metrics JSON | `verdicts/task336_issue49_stage4_beam20.json` |
| Issue #50 真实噪声测量（同期） | `verdicts/task337_issue50_method_a_b_residual.json` |

---
*Commit + push + Issue #49 close to follow R15.*
