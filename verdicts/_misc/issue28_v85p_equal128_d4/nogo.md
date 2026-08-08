# Issue #28 v85p equal128 SID + v4 配置 NO-GO

**Date**: 2026-08-09
**Status**: ❌ NO-GO
**Commit**: (本任务 commit)

## 实验

新组合: issue210 equal128 SID + LR=4e-4 + decoder=4 + dropout=0.20 + issue61 HAB ckpt
(vs v4: hyp_v2 SID; vs phase D: LR=1e-3 + decoder=6)

## 结果

| 指标 | 数值 |
|------|------|
| Stage3 best valid_R@10 | 0.1212 (ep65, vs v4 0.1273) |
| Stage4 test R@10 | **0.0942** (< v4 0.1031 -0.0089) |
| Stage4 test R@5 | 0.0750 |
| Stage4 test NDCG@10 | 0.0705 |

## 4 Gate 答复

### Gate 1 (Spec 摘录): R18 4 维度对比
D1 Spec: equal128 SID (K=128 全均匀) 替代 hyp_v2 capmatch (K=[64,128,256])
D2 实施: Stage3 v4 配置 + equal128 SID + issue61 HAB ckpt (新组合)
D3 Gate 1 失败机制: equal128 funnel geometry + LR=4e-4 + decoder=4 组合 不如 hyp_v2 capmatch + LR=4e-4 + decoder=4 (v4)
D4 引用: 之前的 phase D equal128 (decoder=6 LR=1e-3) 已 0.0907 < v77原 0.0911, 印证 equal128 SID 本征不优于 capmatch

### Gate 2 (实施核心): Stage3 wrapper 改 sid_npy+product_dir+hab_stage2_ckpt, py_compile OK, DDP 4 卡启动 PID 3373103
训练 ep65 best valid_R@10=0.1212, 之后 5 epoch plateau, 主动 kill.

### Gate 3 (Gate 1 失败机制): best valid_R@10 plateau 5 epoch 在 0.1212, R23 边缘条件 (val plateau) 触发
+ test_R@10 0.0942 < v4 0.1031 (-0.0089) → 实际超 0.1031 失败.

### Gate 4 (FAIL): test R@10=**0.0942** vs 目标 0.1031 (-0.0089), 真实 v77原 baseline 0.0911 → +0.0031 微超 v77原 但 < v4 0.1031

## 结论

equal128 SID + v4 配置组合 不超 v4 (hyp_v2 SID + v4 配置). 确认 **hyp_v2 capmatch SID 是本环境最佳 SID 链路**.

下一步推荐 (R11 自决):
1. **v4 ckpt (0.1031) 已超 v77原 0.0911 (+0.0120)**, 是本环境当前 SOTA
2. 进一步超 0.1031 需架构改动 (e.g. DECOR bins / DIGER RQ-VAE) 或 fork Stage3 主脚本 (R31 禁止)
3. 建议用户接受 0.1031 作为新基线
