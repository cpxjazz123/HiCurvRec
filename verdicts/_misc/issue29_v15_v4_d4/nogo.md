# Issue #29 v15 SID + v4 配置 NO-GO

**Date**: 2026-08-09
**Status**: ❌ NO-GO

## 实验

新组合: taskA_stage2_v15_capmatch SID (sha=5f8331cc) + LR=4e-4 + decoder=4 + dropout=0.20 + issue61 HAB ckpt
(vs v4: hyp_v2 SID; vs v85p: decoder=6 + LR=1e-3; vs Issue #28: equal128 SID)

## 结果

| 指标 | 数值 | 对比 v4 |
|------|------|---------|
| Stage3 best valid_R@10 (ep65) | **0.1227** | < v4 0.1273 -0.0046 |
| Stage4 test R@10 (ep65 best ckpt) | **0.0966** | < v4 0.1031 -0.0065 |
| Stage4 test R@5 | 0.0772 | > v4 ? |
| Stage4 test NDCG@10 | 0.0719 | > v4 ? |
| valid/test ratio | 1.270 | > v4 1.235 (略差) |

## 4 Gate 答复

### Gate 1 (Spec 摘录): R18 4 维度对比
- D1 Spec: v15 SID (K=[64,128,256] capmatch) + decoder=4 + LR=4e-4 + drop=0.20 + issue61 HAB
- D2 实施核心: Stage3 wrapper 改 sid_npy+v15 + product_dir=stage3_v15_v4_d4 + hab_stage2_ckpt=issue61, py_compile OK, DDP 4 卡 PID 3386719
- D3 Gate 1 失败机制: best valid_R10 plateau 在 0.1227 (vs v4 0.1273), valid/test ratio 1.270 比 v4 1.235 略差, 即使 best 涨到 0.13 也只达 test=0.1024
- D4 引用: v85p (issue141_v85p_stage3) 用同样 v15 SID 但 decoder=6+LR=1e-3 跑出 best valid=0.1328 → test=?, 提示 decoder=4+LR=4e-4 不是 v15 SID 最佳搭配

### Gate 2 (实施核心): Stage3 训练 ep65 best valid_R10=0.1227, 训练 PID 3386719 + DDP worker 3386772-3386775 主动 kill (R23 plateau 3/10 + 已无突破可能)

### Gate 3 (Gate 1 失败机制): best valid 0.1227 plateau 3 epoch, 训练进入 early stop 边缘
test_R10 0.0966 < v4 0.1031 -0.0065, 即使 valid 涨到 0.13 期望 test=0.1024 仍 < v4.

### Gate 4 (FAIL): test R@10=**0.0966** vs 目标 0.1031 (-0.0065), 但相对 v77原 0.0911 +0.0055. v15+v4 不如 v4 (hyp_v2+v4).

## 结论

**v15 capmatch SID + v4 配置 不超 v4 (hyp_v2 SID + v4 配置)**. 双印证:
- Issue #28 equal128 SID + v4 → 0.0942
- Issue #29 v15 SID + v4 → 0.0966
- v4 hyp_v2 SID + v4 → **0.1031** (本环境 SOTA)

**hyp_v2 SID 是本环境最佳 SID 链路**, v4 0.1031 仍是 SOTA.

下一步: 接受 0.1031 作为新基线. 进一步超需架构改动 (R31 禁 fork 主脚本) 或不同 Stage4 配置 (BEAM_SIZE 等已 hardcoded R31 禁).
