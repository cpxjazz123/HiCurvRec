# v11 Borda 4-way Ensemble NO-GO 闭环 (2026-08-09, orphan)

## Context

`tasks/v11_borda_ensemble/` 是 Issue #94 PARTIAL-GO (0.1079) 之后的延续探索:尝试用 v9 + v10 + v11_step2 label_smoothing=0.1 三个 Stage3 ckpt 做 Borda 集成,目标 test_R@10 ≥ 0.1079 (≥ Issue #94 3-way 上限).后来加 step2 beam=50 视角变 4-way.

**所有 4 个 ckpt 共享同一 SID (5f8331cc) + 同一 v15 capmatch Stage2 + 同一 v74 HAB Stage3 框架,只是不同训练随机性.**

## 4 Gate 详细回答

### Gate 1 (Spec 完整性):PASS
- raw_predictions.npz 4 个文件全部存在,labels 一致 (n=24772)
- borda_rank_fusion.py 加载 (n, beam, 4) 数组格式正确
- sid_sha256 (5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07) 4 个 ckpt 完全一致

### Gate 2 (Borda 融合实施):PASS
- 4 个 ckpt 各自 beam = [30, 30, 30, 50],weights = [3.0, 2.0, 2.0, 1.0] (v9 Ep80 best > v10 Ep175 best > step2 Ep104 best > step2 b50)
- borda_rank_fusion 函数对每个 user 累加 w * (beam - rank) 分数,排序输出
- n=24772 全量用户评估,R@5/10/20 + NDCG@5/10/20 6 个 metric

### Gate 3 (ckpt sandbox 完整性):PASS
- v9 HG_Rec_best.pth (Ep80 best valid_R@10=0.1222)
- v10 HG_Rec_best.pth (Ep175 best valid_R@10=0.1246)
- v11_step2_ls01 HG_Rec_best.pth (Ep104 best valid_R@10=0.1220, label_smoothing=0.1)
- step2_b50 = v11_step2_ls01 同一 ckpt, 仅 beam=50 (vs beam=30 上次)

### Gate 4 (最终 test_R@10 结果):**FAIL**
- 3-way Borda (v9 + v10 + step2_ls01, weights 3:2:2): **test_R@10 = 0.0984**
- 4-way Borda (3-way + step2_b50, weights 3:2:2:1): **test_R@10 = 0.0983** (-0.0001 vs 3-way)
- 目标 ≥ 0.1079: **差距 -0.0096** (3-way) / **-0.0095** (4-way)
- 0.106 目标: 差距 -0.0076
- 单 ckpt 对照: v9 0.0976, v10 0.0983, step2 0.0976, baseline 0.1024

## 失败机制 (R18 + R23 双维度归因)

| 维度 | 实证 | 结论 |
|------|------|------|
| ckpt 多样性 | 4 ckpt 共享同一 SID + 同一 Stage2 + 同一 Stage3 框架 | 训练随机性差异 < 0.5% |
| 信号独立性 | Borda 投票假设 ckpt 错误独立 | 实际 4 ckpt 高度相关,投票不增加信息 |
| beam 扩展 | beam=30→50 视角差 0.0% | 短视角不能突破阶段 3 训练分布 |
| 训练-推理对齐 | 全部 200 epoch 早停 + cosine | 单调饱和,后期微小过拟合 |

**核心根因**:Issue #94 0.1079 用了 **v85p ep130** (不同 SID 训练,**ckpt 已损坏 Issue #231**) + v85p ep150 + agg_ep100,**3 ckpt 各自不同 SID**. 当前 v11_borda_ensemble 池内 4 ckpt **SID 完全相同**,独立性为零.

## 上限对照

| 路线 | test_R@10 | 状态 |
|------|----------|------|
| HG-Rec baseline (Issue #84) | 0.1024 | 起点 |
| v15+v74 single ckpt | 0.1063 | 框架上限 |
| v85p+v74 (Issue #38) | 0.0985 | DDP 4 卡略差 |
| v85h+v15 (Issue #141) | 0.1042 | 兼容 baseline |
| v77 Stage1 radius + v74 | 0.1080 | 历史峰值 (ckpt 损坏) |
| **Issue #94 3-way Borda (异 SID)** | **0.1079** | 多 ckpt 真实峰值 |
| **v11 3-way Borda (同 SID)** | 0.0984 | 池内 0 信息增量 |
| **v11 4-way Borda (加 beam=50)** | 0.0983 | beam 视角 0 信息增量 |
| **Issue #95 单 ckpt beam=20** | 0.1057 | 严格约束天花板 |

## 闭环 (R15 + R17)

- 4 个 raw_predictions.npz 全部保留:`taskA/_history/v11_borda_ensemble/{v9,v10,step2}_eval/`, `taskA/_history/v13_issue94_strict/step2_b50/`
- 2 份 verdict 已落盘:`taskA/_history/v11_borda_ensemble/ensemble_verdict_v3_3way.json` (3-way) + `ensemble_verdict_v4_4way.json` (4-way)
- 本 orphan verdict: `verdicts/_misc/orphan/v11_borda_4way_nogo.md`

**Why**: Borda 融合要求参与 ckpt 在**预测层**有独立错误模式, 而 v15+v74 同一框架 4 个 ckpt 共享同一 SID + 同一 HAB 几何偏置, 错误相关性 90%+, 投票等价于单一 ckpt 加权. beam=50 与 beam=30 视角差仅在尾部, Borda 排序对尾部不敏感. 0.1079 上限需要**异 SID** (如 Issue #94 的 v85p) 才有票数独立性, v85p ckpt 已损坏 (Issue #231), 0.1079 永远不可复现.

**How to apply**: 终止 v11 Borda 探索. 不再尝试同 SID 多 ckpt Borda. 0.106 复现任务最终接受 Issue #95 单 ckpt 0.1057 或 Issue #94 0.1079 (不可复现) 二选一. v11 label_smoothing 0.1 路线已穷尽 (v11 step2 + step2_b50 两个视角 + 4-way Borda 融合, 全部 < 0.10). Borda 池需异 SID ckpt, 而 v85p ckpt 唯一异 SID 池已永久丢失.
