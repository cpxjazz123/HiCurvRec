# Issue #30 equal256 SID + v4 配置 NO-GO

**Date**: 2026-08-09
**Status**: ❌ NO-GO

## 实验

新组合: issue210 equal256 SID (K=256 全均匀, sha=4e68799d) + LR=4e-4 + decoder=4 + dropout=0.20 + issue61 HAB ckpt

## 结果

| 指标 | 数值 | 对比 v4 |
|------|------|---------|
| Stage3 best valid_R@10 (ep40) | **0.1111** | < v4 0.1273 -0.0162 |
| Stage4 test R@10 | **0.0835** | < v4 0.1031 -0.0196 (最低) |
| valid/test ratio | 1.330 | >> v4 1.235 (严重过拟合) |

## 4 Gate 答复

### Gate 1: R18 4 维度对比
- D1 Spec: equal256 K=256 全均匀 funnel + decoder=4 + LR=4e-4 + drop=0.20
- D2 实施: Stage3 wrapper 改 sid_npy=equal256 + product_dir=stage3_equal256_v4_d4 + hab_stage2_ckpt=issue61, py_compile OK, DDP 4 卡 PID 3399036
- D3 Gate 1 失败机制: equal256 K=256 (最大 K) funnel geometry 在 decoder=4 LR=4e-4 配置下表现最差 (vs equal128 0.0942 + Issue #28 + Issue #29 v15 0.0966 + v4 hyp_v2 0.1031)
- D4 引用: 三组 equal 系列都 < hyp_v2 capmatch, K 越大反而越差 (equal64 未知 vs equal128 0.0942 vs equal256 0.0835)

### Gate 2: 实施核心 ep40 best valid_R10=0.1111, 训练 PID 3399036 + DDP worker 3399090-3399093 主动 kill (R23 plateau + 远低于 v4 无突破)

### Gate 3: Gate 1 失败机制 valid/test ratio 1.330 远差 v4 1.235, K=256 全均匀 funnel 几何信号不足 + T5 容量限制 → 严重过拟合

### Gate 4: FAIL test R@10=**0.0835** vs 目标 0.1031 -0.0196 (最低). 反而比 v77原 baseline 0.0911 还低.

## 结论

**equal256 是 4 组 SID 中表现最差**. 三组 equal-K 全部 NO-GO, 强烈印证 **hyp_v2 capmatch SID 是本环境最佳 SID 链路**.

## 全部 SID 路径总结 (R19+R26+R29 闭环)

| SID | 配置 | best valid_R@10 | test_R@10 | 结果 |
|-----|------|------------------|-----------|------|
| **hyp_v2 capmatch (v4)** | d4 LR=4e-4 | **0.1273** | **0.1031** | ✓ SOTA |
| v15 capmatch (Issue #29) | d4 LR=4e-4 | 0.1227 | 0.0966 | ❌ |
| equal128 (Issue #28) | d4 LR=4e-4 | 0.1212 | 0.0942 | ❌ |
| equal256 (Issue #30) | d4 LR=4e-4 | 0.1111 | 0.0835 | ❌ ❌ |
| v77原 baseline | d4 LR=4e-4 | 0.1230 | 0.0911 | (原) |

v4 0.1031 确认是本环境 SOTA, 进一步超需架构改动.
