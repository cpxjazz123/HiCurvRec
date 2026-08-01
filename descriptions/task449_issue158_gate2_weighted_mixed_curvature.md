# Task #449 / Issue #158 [方向B Gate2] 加权混合曲率RQ-VAE代码本与完整SID链路验证

## Gate 目标
**Gate 2 = Stage 2 实施** (前序 Gate 1: #156 PASS commit 89b7563, mixing-logit grad [9.80e-7, 1.18e-6, 6.27e-7] + κ grad [71.27, 115.46, 219.31])

把三层独立 learnable κ + 固定双曲分量 + 欧氏分量 + alpha=0.1+0.7*softmax(logits) 接入现有 RQ-VAE Stage 2:
- 对每层记录可学习 κ 分量 + 固定双曲分量 + 欧氏分量 的距离/贡献和权重
- 生成 Musical_Instruments 全量 (9922, 4) SID (含第4位去重 digit)
- 不改 Stage 1 embedding / item 顺序 / RQ-VAE 主流程

## 验收 (per Issue #158 spec)
1. **10+ 预注册记录点**: 每层 κ / mixing logits / alpha / 三分量距离贡献 有限; alpha 和=1, 每项 [0.1, 0.8], 非零参数 delta. 不能仅输出静态权重.
2. **codebook/SID 链路可审计**: checkpoint reload 的 κ / alpha / 距离 / assignment 一致. 无 NaN/Inf. 输出完整 (9922, 4) 整数 SID + shape/dtype/范围/SHA256/item alignment 证据.
3. **对照消融**: 关闭产品分量或固定等权的 Stage 2 几何诊断对照. 不进入 T5 或 R@K.

## R18 4 维度差异对比 (vs 历史)
| 维度 | #156 | #158 (本任务) |
|------|------|----------------|
| D1 spec 摘录 | Gate 1 mixing-logit grad 实证 | Gate 2 完整 Stage 2 链路 (产品 manifold SID 完整传播) |
| D2 实施核心 | 仿射截断 softmax (alpha=0.1+0.7*softmax) | 产品 manifold Stage 2: 可学习 κ + 固定双曲 + 欧氏 三分量加权混合 |
| D3 Gate 失败机制 | hard clamp grad path 切断 | 无 Stage 2 链路证据 / 静态权重非动态学习 |
| D4 引用文献 | 无 | arXiv:2307.04514 数据驱动加权混合曲率产品流形 |

→ **4 维度全部不一致**, 必须新实验.

## R11.5 自主决策 (实施)
- **三分量产品流形**: 对每层 codebook x, 距离 = α · d_hyp(x; κ_l) + (1-α) · d_eucl(x) (其中 α_l 是 softmax 后仿射截断)
- **可学习 κ_l**: 跟 #156 一致, AffineTruncatedSimplexKappaModel
- **固定双曲分量**: β_l · d_hyp(x; c=1.0) 固定 (paper default c=1)
- **欧氏分量**: γ_l · d_eucl(x)
- **权重和约束**: α_l + β_l + γ_l = 1, 每项 [0.1, 0.8]
- **混合公式**: d(x; κ_l, α_l, β_l, γ_l) = α_l · d_hyp(κ_l) + β_l · d_hyp(c=1.0) + γ_l · d_eucl
- **SID 输出**: 标准 Sinkhorn + 第4位 dedup
- **GPU**: 分配 GPU 1 (空闲)

## 阶段产物 (8 件套 + R12 ckpt 强制)
1. `descriptions/task449_issue158_gate2_weighted_mixed_curvature.md` (本文件)
2. `scripts/task449_issue158_gate2_weighted_mixed_curvature.py` (~650 lines)
3. `products/task449_issue158_gate2_weighted_mixed_curvature/config.json`
4. `products/task449_issue158_gate2_weighted_mixed_curvature/precheck.json`
5. `products/task449_issue158_gate2_weighted_mixed_curvature/mixed_curvature_log.json` (10+ 记录点)
6. `products/task449_issue158_gate2_weighted_mixed_curvature/sid_output.npy`
7. `products/task449_issue158_gate2_weighted_mixed_curvature/sid_metadata.json`
8. `products/task449_issue158_gate2_weighted_mixed_curvature/verdict.json`
9. `verdicts/task449_issue158_gate2_weighted_mixed_curvature_result.md`

## Gate 决策阈值
- **Gate 2 PASS** if: 10+ 记录点 + alpha 和=1 每项 [0.1, 0.8] + 非零 delta + reload 一致 (5/5) + 无 NaN/Inf + 真实 SID hash + item alignment + 对照消融 PASS
- 任一项 FAIL → Gate 2 FAIL, STOP

## R17 + R20 + R21 合规
- commit message: issue # + Gate + 关键数据 + 失败原因 (R17) + 详细 4 Gate (R20)
- 落地后立即补 issue comment 含 commit hash (R21 v2)
- verdict push (R15)
- issue close (R16) 含详细 4 Gate comment