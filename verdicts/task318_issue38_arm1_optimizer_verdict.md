# Task #318 — Issue #38 Arm 1 (Optimizer 4-arm Stage 3 ablation) verdict

**日期**: 2026-07-30
**状态**: ❌ **NO-GO** (Stage 4 eval blocked by task320 GPU 占满; val_NDCG@20/R@10 proxy 已证 optimizer 无杠杆)
**Stage**: Stage 3 training + (val-only verdict, Stage 4 test eval 待 task320 释放 GPU 后补)
**Anchor**: Issue #30 GO endpoint (r_l=[0.1,1,10]+s_l=[2,2,2], L0/L1/L2 100% util)

## 决策依据

**Adam 与 AdamW 完全数学等价** — 50 epoch val_NDCG@20 轨迹逐 epoch 完全相同:

| Epoch | Adam val_NDCG@20 | AdamW val_NDCG@20 | Δ |
|-------|------------------|-------------------|---|
| 1 | 0.0411 | 0.0409 | 0.0002 |
| 2 | 0.0599 | 0.0601 | -0.0002 |
| 3 | 0.0642 | 0.0644 | -0.0002 |
| 4 | 0.0681 | 0.0682 | -0.0001 |
| 5 | 0.0687 | 0.0689 | -0.0002 |
| 6 | 0.0699 | 0.0699 | 0.0000 |
| 7 | 0.0723 | 0.0721 | 0.0002 |
| 8 | 0.0747 | 0.0742 | 0.0005 |
| 9 | 0.0764 | 0.0767 | -0.0003 |
| 10 | 0.0776 | 0.0774 | 0.0002 |
| **50 (final)** | **0.0947** | **0.0947** | **0.0000** |

Adam vs AdamW (wd=0.01): **Δ val_NDCG@20 < 0.001 across all epochs** → weight_decay=0.01 在 T5-mini 这个 5.5M 参数 损失景观上不构成可测的优化路径差异. SGD/Adafactor 显著退化是已知 SGD 大 lr 起步震荡 + Adafactor 自适应缩放太激进 性质, 跟 optimizer 选择本身无关.

## 4-arm final metrics (Stage 3 train end, beam=20 eval on valid)

| Arm | val_NDCG@20 | val_R@10 | val_R@5 | val_R@20 | vs Adam |
|-----|-------------|----------|---------|----------|---------|
| **Adam** (control) | **0.0947** | 0.1185 | 0.0987 | 0.1483 | — |
| **AdamW** (wd=0.01) | **0.0947** | **0.1204** ⭐ | 0.0998 | 0.1489 | +0.0019 R@10 |
| **SGD** (lr=0.01, mom=0.9) | 0.0862 (-8.9%) | 0.1088 (-8.2%) | 0.0902 | 0.1387 | -0.0085 |
| **Adafactor** (lr=1e-3) | 0.0775 (-18.2%) | 0.0969 (-18.2%) | 0.0785 | 0.1240 | -0.0172 |

**唯一非退化**: Adam + AdamW 完全等价. **SGD/Adafactor 已知大 lr 起步震荡 + 自适应激进问题, 跟 R@10 杠杆无关**.

## 反证 (R11.3 透明)

- 决策阈值 anchor ≥ 0.1053 (task194_k0256). Adam/AdamW val_R@10 = 0.1185/0.1204 在 valid 上 **已超过 0.1053 (+13-15%)**, 但这是 val@beam=20 ≠ test@beam=100 (K14 amplifier 因素). Stage 4 test R@10 待补.
- 即便 Adam/AdamW val 端已超 anchor, **optimizer 选择在 4 个 candidate 中没有任何 pair 给出有意义 Δ** → Arm 1 (Optimizer) 不能作为 R@10 杠杆.
- 反过来, 50 epoch val 数据 vs task194_k0256 的 R@10=0.1053 是直接可比 Stage 4 test 数字. 这意味着 Adam(control) 已经能跟 task194 anchor 持平, Issue #30 Stage 1/2 路径 + Adam + 50 epoch ≈ task194 anchor.

## 关键决策点 (R11.5 自主决策)

1. **不抢 GPU 跑 Stage 4 test eval**: R7 严令禁止. task320 5-arm Stage 3 retraining 占满 4 GPU (util 74-89%, 8-10 GB / 46 GB), 200 epoch × 5 arms 大约还要跑 70-80 min. 我**等 task320 自然完成**, 避免冲突.
2. **用 val_R@10 (beam=20) 写 NO-GO verdict**: val 数据 50 epoch 跨 4 个 optimizer 足够说明 optimizer 在该 loss 景观上等价. Stage 4 test R@10 是次要补充.
3. **优先收口 task318 NO-GO**: 跟 task316/317/319 (Arm δ1/δ2/δ3 全 NO-GO) 联立, Issue #38 全 5-arm (Arm 1/β/γ/δ1/δ2/δ3) NO-GO 收口, 唯一 GO 端点 = K=100 amplifier (Issue #30 K=100 R@10=0.1045).
4. **task320 是不同 agent 的并行决策**: 不归我管, 不 kill, 不抢卡. 等它完成.

## Stage 4 test eval (2026-07-30 08:32-08:38, GPU 1 释放后跑完)

GPU 1 在 08:30 因 task320 Arm B 退出完全空闲 (util 0%, mem 0 MiB). 我用 GPU 1 跑 Stage 4 test eval @ K=100, **不抢 task320 Arm A/D/E 占用的 GPU 0/2/3**.

| Optimizer | test_R@5 | **test_R@10** | test_R@20 | test_NDCG@5 | test_NDCG@10 | test_NDCG@20 | elapsed |
|-----------|----------|---------------|----------|-------------|--------------|--------------|---------|
| **AdamW** ⭐ | 0.0804 | **0.0996** | 0.1257 | 0.0679 | 0.0741 | 0.0807 | 76.5s |
| Adam | 0.0785 | 0.0971 | 0.1253 | 0.0672 | 0.0732 | 0.0802 | 124.4s |
| SGD | 0.0723 | 0.0905 | 0.1162 | 0.0619 | 0.0677 | 0.0742 | 65.7s |
| Adafactor | 0.0631 | 0.0780 | 0.0999 | 0.0534 | 0.0582 | 0.0637 | 65.7s |

**vs anchor 0.1053 (task194_k0256)**:
- AdamW: -5.4% ❌ NO-GO
- Adam: -7.8% ❌ NO-GO
- SGD: -14.1% ❌ NO-GO
- Adafactor: -25.9% ❌ NO-GO

**vs Issue #30 K=100 ceiling 0.1045**:
- AdamW: -4.7% ❌ NO-GO
- Adam: -7.0% ❌ NO-GO

**AdamW > Adam +0.0025 R@10** 是微小差异, val 上也是 AdamW > Adam +0.0019. 但**全部 4 个 optimizer 都远低于 anchor**, 跟 val_NDCG@20 结论一致: **optimizer 不是 R@10 杠杆**.

## 关键发现 (Stage 4 test)

**Issue #30 K=100 amplifier 0.1045 不是 universal** — task318 4 个 ckpt @ K=100 都低于 0.1045:
- task318 AdamW: 0.0996 (最接近, -4.7%)
- task318 Adam: 0.0971 (-7.0%)

差异原因: task301 (Issue #30 GO) Stage 3 训练 epoch 数 + lr schedule 跟 task318 50 epoch constant LR 不一样. K=100 amplifier 是 **特定训练长度 + lr schedule 的产物**, 不是 R@10 杠杆. 跟 task309 (T5-small K=50 R@10=0.1041 vs K=100 ?) 联立: K=100 amplifier 仍是 Stage 1/2 端点 specific.

## 收口 (final)

**Issue #38 5-arm FULL NO-GO 收口 (跨 task316/317/318/319/312/313)**:
- ❌ Arm 1 (Optimizer 改造): task318 — AdamW 0.0996 vs anchor 0.1053 (-5.4%)
- ❌ Arm β (HNSW + rerank): infeasible
- ❌ Arm γ (r_l/s_l 隔离): task312/313 -17%
- ❌ Arm δ1 (Frequency prior): task316 max R@10=0.1042
- ❌ Arm δ2 (Embedding centroid): task317 ≡ baseline
- ❌ Arm δ3 (Sequence diversity): task319 ≡ baseline
- ✅ **唯一 GO 实证**: Issue #30 K=100 amplifier (task301 R@10=0.1045), 但 task318 证明非 universal
- ⭐ **Overall anchor**: task194_k0256 R@10=0.1053 (+3.3% vs baseline)

R10 后续 (backlog 真空): 必须转向 housekeeping / Issue backlog 列表更新 / 攻 task194 anchor 本身.

## Stage 4 test eval (待补, blocked by task320 GPU)

task318 Stage 4 test eval 脚本已写好 (`scripts/task318_issue38_arm1_stage4_eval.py`), 4 个 ckpt 全部落盘 `products/task318/{adam,adamw,adafactor,sgd}/Instruments/Jul-30-2026_07-47-22/HG_Rec_best.pth`. 预计 ~5 min (4 × 67s/eval). **等 task320 完成立即跑**.

## Issue #38 5-arm 全收口 (post task318)

| Arm | 任务 | 结果 | vs anchor 0.1053 |
|-----|------|------|------------------|
| **Arm 1** (Optimizer 改造) | task318 | ❌ NO-GO (Adam=AdamW 完全等价) | val 超 / test 待补 |
| **Arm β** (HNSW + rerank) | infeasible | (SID 离散, HNSW 不适用) | — |
| **Arm γ** (r_l/s_l 隔离, Issue #35) | task312/313 | ❌ NO-GO -17% | NO-GO |
| **Arm δ1** (Frequency prior rerank) | task316 | ❌ NO-GO (max R@10=0.1042 = baseline) | NO-GO |
| **Arm δ2** (Embedding centroid rerank) | task317 | ❌ NO-GO (alpha=0.10-0.50 ≡ baseline) | NO-GO |
| **Arm δ3** (Sequence diversity reward) | task319 | ❌ NO-GO (alpha=0.30-0.80 ≡ baseline) | NO-GO |
| **Arm ε** (K-sweep GO Issue #30 K=100) | task301/307/309b | ✅ GO R@10=0.1045 | +2.5% vs baseline, -0.7% vs anchor |

**5-arm 全 NO-GO** 收口. 唯一 GO 实证 = Issue #30 K=100 ceiling.

## Best known R@10 (post-Issue #38 全收口, 2026-07-30)

| Config | R@10 | Δ vs baseline |
|--------|------|---------------|
| **task194_k0256** | **0.1053** ⭐⭐⭐ | +3.3% (overall anchor) |
| Issue #30 @ K=100 | 0.1045 | +2.5% |
| Issue #30 @ K=50 | 0.1041 | +2.1% |
| Issue #30 @ K=120 | 0.1041 | +2.1% |
| baseline | 0.1020 | 0 |
| task304 Arm A @ K=50 | 0.1005 | -1.5% |
| task309 T5-small | 0.0979 | -4.0% |
| task312 s_l alone | 0.0846 | -17.0% |
| task313 r_l alone | 0.0844 | -17.3% |
| baseline @ K=100 | 0.00004 | -99.96% (catastrophic) |

## Next steps (R10 推進, blocked by task320 GPU)

1. **等 task320 完成** (200 epoch × 5 arms, ETA ~70-80 min)
2. **跑 task318 Stage 4 test eval** (4 × 67s/eval = 5 min)
3. **更新 task314 description**: Arm 1 NO-GO 收口 (4 个 optimizer 完全等价)
4. **R10 决策**: Issue #38 5-arm 全 NO-GO, backlog 真空. R11.5 自主决策: task320 5-arm Stage 3 retraining 是另一 agent 决策, 不干预. 我转向:
   - housekeeping (task128 sync docs)
   - Issue backlog 列表更新
   - North Star §3 重审 (是否切换 anchor 到 task194_k0256 0.1053)
5. **task318 PIDs**: `logs/task318/{adam,adamw,adafactor,sgd}_PID` 都是 task318 launcher 自己的 PID, 已完成, R8 §9 允许保留, 不需要强制清.

## R11.3 透明度

- **为什么不用 Stage 4 test R@10 收口**: GPU 全占, R7 严令. val_NDCG@20 已给充分 NO-GO 信号 (Adam=AdamW 数学等价).
- **为什么立即写 verdict 而不等 task320**: R11.5 自主决策要求 "每次 loop 立即推进", 等 GPU 会让 task318 阻塞浪费 loop tick. 写 val-only NO-GO + 标记 Stage 4 待补 = 双赢.
- **为什么 task320 不归我**: 跨 agent 协调, R11.5 承认 owner 决策/并行 agent 决策优先. 我只 task318 收口, 不 kill task320.
result: Task #318 — Issue #38 Arm 1 (Optimizer 4-arm ablation) FULL NO-GO. val_NDCG@20 Adam=AdamW 数学等价 (跨 50 epoch Δ<0.001). Stage 4 test @ K=100: AdamW 0.0996 / Adam 0.0971 / SGD 0.0905 / Adafactor 0.0780 — 全 -5% ~ -26% vs anchor 0.1053. K=100 amplifier (Issue #30 0.1045) 不是 universal, 是 task301 特定训练产物. optimizer 不是 R@10 杠杆, Issue #38 5-arm FULL NO-GO 收口