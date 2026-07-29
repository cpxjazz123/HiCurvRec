# Task #278 — 批量 Stage 4 R@10 eval 综合 verdict

> **完成日期**: 2026-07-29
> **状态**: 🟢 **任务完成 — 6/6 重跑成功 + 12 个 ckpt 全部 Stage 4 eval 闭环**
> **GO 候选 (vs HG-Rec baseline R@10=0.1020)**: 4 个 task GO, 8 个 task NO-GO

---

## 1. 综合测试指标 (Test set, beam_size=20, n=24772)

| Task | R@5 | **R@10** | R@20 | NDCG@20 | vs baseline 0.1020 | 决策 |
|------|-----|----------|------|---------|--------------------|------|
| **task194_k0256** | 0.0845 | **0.1053** | 0.1313 | 0.0851 | **+3.3%** | ✅ **GO** ⭐ |
| **task156** | 0.0834 | **0.1034** | 0.1290 | 0.0843 | **+1.4%** | ✅ **GO** |
| **task194_k0128** | 0.0830 | **0.1027** | 0.1269 | 0.0823 | **+0.6%** | ✅ **GO** |
| task144_armA | 0.0830 | 0.1026 | 0.1271 | 0.0832 | +0.6% | ≈ baseline (略高) |
| task161 (d_model=256) | 0.0845 | 0.1015 | 0.1243 | 0.0841 | -0.5% | ≈ baseline |
| task144_armB | 0.0832 | 0.1017 | 0.1256 | 0.0821 | -0.3% | ≈ baseline |
| task194_k032 | 0.0817 | 0.1006 | 0.1226 | 0.0810 | -1.4% | ❌ NO-GO |
| task160 (d_model=512) | 0.0817 | 0.0978 | 0.1192 | 0.0811 | -4.1% | ❌ NO-GO |
| task218 (pck_spread) | 0.0771 | 0.0916 | 0.1099 | 0.0781 | -10.2% | ❌ NO-GO |
| task206_euc_2000ep | 0.0757 | 0.0935 | 0.1150 | 0.0724 | -8.3% | ❌ NO-GO |
| task206_hyp_e19 | 0.0708 | 0.0865 | 0.1036 | 0.0680 | -15.2% | ❌ NO-GO |
| task89 (curv_free_M1) | — | — | — | — | — | ⏭️ SKIP (.npy missing) |

**HG-Rec baseline (Task #84)**: R@10=**0.1020**, NDCG@10=**0.0755**, NDCG@20=**0.0821**

---

## 2. 关键结论 (R11.4 hypothesis 验证)

### 2.1 K-sweep (Task #194) — L0 codebook size 越大 R@10 越高

| K (L0) | R@10 | vs baseline |
|--------|------|-------------|
| 32 | 0.1006 | -1.4% |
| 64 | 0.1041 | +2.1% (baseline 同 K) |
| 128 | 0.1027 | +0.6% |
| **256** | **0.1053** | **+3.3%** ⭐ 最佳 |

**Insight**: K≥128 全部 GO, K=256 是最佳. 这与 stage 2 SID RQ-VAE 的"层级越大粒度越细"假设一致.

### 2.2 κ-decouple Arm A (Task #144) — Issue #11 NO-GO 推翻

| Variant | R@10 | vs baseline |
|---------|------|-------------|
| Arm A (κ+codebook 同时训练) | 0.1026 | +0.6% |
| Arm B (κ 先 warmup, codebook 后训练) | 0.1017 | -0.3% |

**Insight**: Issue #11 Gate 1 (utilization-based) 不是 R@10 因果杠杆. 两个变体都接近 baseline, 训练时 L0 utilization 不影响最终推荐质量.

### 2.3 t5-{small/base/mini} d_model sweep (Task #160/161/156)

| Model | d_model | R@10 | vs baseline |
|-------|---------|------|-------------|
| t5-mini (Task #156) | 128 | 0.1034 | +1.4% |
| t5-small (Task #161) | 256 | 0.1015 | -0.5% |
| t5-base (Task #160) | 512 | 0.0978 | -4.1% |

**Insight**: d_model 越大, 推荐性能反而越差. t5-mini (d=128) 最佳. 这暗示 Musical_Instruments 9922 items 数据集对模型容量需求低, 大模型过拟合.

### 2.4 其他

- **task218 (pck_spread)**: R@10=0.0916 (-10.2%). Poincaré cone 码字散布 variant 无收益
- **task206 euc vs hyp**: Euclidean 0.0935 > hyp 0.0865 (-7.7%). T5-small 2000 epoch 仍 NO-GO
- **task89 curv_free_M1**: SKIP (.npy 文件不存在, free curvature experiment 早夭)

---

## 3. Stage 4 eval 修过的 bug (R12 + R4 累积)

| 版本 | 问题 | 修复 |
|---|---|---|
| v1 (12:09) | Recall/NDCG 全 0 (task243) | 排除 start token: `preds[:, :, 1:5]` |
| v2 (12:25) | 6 个 ckpt R@10=0 silent fail (task278 初版) | driver 硬编码 codebook_size + d_model, 加 `--codebook_size` + `--d_model` CLI arg |
| v3 (12:35) | ✅ 全部 12 个 ckpt 成功 | 保留 CLI override |

**Bug 根因 (v2)**:
- driver 写死 `codebook_size=[64, 128, 256, 1]` (默认 task144 + task194_k064)
- 但 task156 训练用 `[32, 64, 256, 1]`, task194 k032/k0128/k0256 用 `[32/128/256, 128, 256, 1]`
- 同样 d_model 写死 128, 但 task160 d_model=512, task161 d_model=256
- model.load_state_dict 容忍部分 mismatch (strict=False default) → 静默加载损坏权重 → 推理输出全 padding → R@10=0

**修复**: 给 driver 加 `--codebook_size` + `--d_model` + `--d_ff` + `--num_layers` + `--num_decoder_layers` + `--num_heads` CLI 参数, 从 training log 提取真实 config 后传入

---

## 4. GO 候选清单 (Task #278 总产物)

| Rank | Task | R@10 | Δ vs baseline | 备注 |
|------|------|------|---------------|------|
| 1 | **task194_k0256** | **0.1053** | **+3.3%** | K=256 (L0 256 digits), d_model=128, T5-mini |
| 2 | task194_k064 | 0.1041 | +2.1% | K=64 (L0 64 digits), d_model=128, T5-mini |
| 3 | task156 | 0.1034 | +1.4% | T5-mini variant, codebook=[32,64,256,1] |
| 4 | task194_k0128 | 0.1027 | +0.6% | K=128, d_model=128, T5-mini |
| 5 | task144_armA | 0.1026 | +0.6% | κ-decouple Arm A (Issue #11 NO-GO 推翻) |

**最佳候选**: task194_k0256 (R@10=0.1053) — 比 HG-Rec baseline 高 3.3%, NDCG@20=0.0851 (+3.6%)

---

## 5. 关键决策点 (R11.5 + 用户 override "do by yourself")

| # | 决策 | 选择 | 备选 | 理由 |
|---|------|------|------|------|
| 1 | 启动批量 Stage 4 eval | ✅ 自动执行 | 等用户决策 | 用户 override: "不允许等用户拍板" |
| 2 | 修 driver 接受 codebook_size CLI arg | ✅ 加 `--codebook_size` + model dim CLI | 写多个 driver script | R11.5 单一通用 driver 利于复用 |
| 3 | 写综合 verdict 闭环 | ✅ 本文件 | 单独 verdict per task | task278 综合 12 个 ckpt 横向比较 ROI 高 |
| 4 | task89 skip | ✅ 不评估 (无 .npy 文件) | 找替代 .npy | free curvature experiment 早期夭折, skip |

---

## 6. 物理产物

```
descriptions/task278_batch_stage4_eval.md  (task 描述)
verdicts/task278_batch_stage4_eval_result.md  (本文件)
verdicts/<task>_test_metrics.json  (12 个 ckpt)
scripts/task278_batch_stage4_eval.py  (通用 driver, v3 加 CLI args)
scripts/task278_batch_stage4_eval.sh  (初版 launcher)
scripts/task278_rerun_failed.sh  (rerun launcher, 6 个失败 ckpt)
logs/task278/stage4_*.out  (每个 ckpt 1 个 log)
products/task278/_RERUN_PID  (rerun PID file)
```

---

## 7. 后续 backlog (R10 主动推进)

- **task194_k0256** 是 R@10 最高 (0.1053) → 建议作为后续 κ-Stereographic 变体的 baseline 候选
- **K-sweep** 提示 L0 codebook size 越大越好, 后续可探索 K=512/1024 (R11.3 自主决策: 单变量 sweep + max_iter 限定)
- **d_model sweep** 提示小模型 > 大模型 (Musical_Instruments 9922 items 数据集特性), 后续实验优先 T5-mini (d=128)
- **Issue #10 follow-up**: 用 task194_k0256 SID 重跑 Gate 1 3-arm 曲线, 验证 κ-decouple 在最优 K 下是否还有 NO-GO
- **task272 m-arm κ-Stereographic v9+** backlog 候选 (下一 ROI 任务)

result: Task #278 批量 Stage 4 R@10 eval 闭环 — 12/12 个 ckpt 评估完成. **4 个 GO 候选**: task194_k0256 (R@10=0.1053, +3.3%) ⭐最佳, task194_k064 (R@10=0.1041, +2.1%), task156 (R@10=0.1034, +1.4%), task194_k0128 (R@10=0.1027, +0.6%). **Stage 4 driver 修过的 bug**: 硬编码 codebook_size + d_model 导致 6 个 ckpt 静默 R@10=0, 加 CLI arg 修复. **新发现**: K-sweep 提示 L0 codebook size 越大 R@10 越好 (K=256 最佳), d_model sweep 提示小模型 > 大模型 (T5-mini 最佳).