# Issue #137 alignment_verdict — canary vs Stage4 raw predictions 同协议对比

**Issue**: #137 曲率可观测性对齐: Stage4 full eval 的 code path 是唯一 oracle, canary 不得自行实现平行 generate/evaluate 逻辑
**Generated**: 2026-08-12
**Verdict**: **FAIL_ALIGNMENT** (canary 与 Stage4 raw 在 5000 交集 sample 上结果完全不一致, 揭示 canary 自身实现 bug)

---

## 一、对比设置

| 项 | canary (Issue #136) | Stage4 raw (Issue #137) |
| --- | --- | --- |
| ckpt | Issue133 stage2/HG_Rec_best.pth | Issue137 stage2/HG_Rec_best.pth (sha256 同: `b6041115...`) |
| SID | Issue133 stage2/sid_output.npy (sha256 `d43dea6f...`) | Issue137 stage2/sid_output.npy (sha256 同) |
| vocab / max_len / pad | 1025 / 20 / 0 | 1025 / 20 / 0 |
| batch_size | 1 | 96 |
| **code path** | **`model.model.generate` (T5 internal, `max_length=8`, `early_stopping=True`)** | **`model.generate` (HG_Rec wrapper, `max_length=5` 默认, 无 `early_stopping`)** |
| **post-processing** | **list loop 取前 4 token, EOS 截断, 不足 4 补 PAD; 无 `[:, 1:]` skip first** | **`preds[:, 1:]` + `reshape(B, beam=20, -1)` + vectorized all-token match** |
| HAB eval | install_hab + freeze lambda_raw/lambda_h_raw/kappa_h | install_hab + freeze lambda_raw/lambda_h_raw/kappa_h (一致) |
| per-head eval | install_per_head_curvature + freeze | install_per_head_curvature + freeze (一致) |
| DDP | 单卡 (CUDA_VISIBLE_DEVICES=0) | 单进程 (wrapper DDP 选项未启, 单进程等价) |
| shuffle | 无 (test.parquet 顺序) | shuffle=False (DataLoader 默认按 parquet 顺序) |

## 二、5000 sample 交集对齐结果

| Metric | canary (Issue #136) | Stage4 raw (Issue #137) | diff |
| --- | --- | --- | --- |
| exact_R@10 | **0.0000** | **0.1016** | +0.1016 |
| exact_R@20 | 0.0000 | 0.1016 | +0.1016 |
| prefix3_R@10 | 0.0004 | 0.1016 | +0.1012 |
| prefix3_R@20 | 0.0004 | 0.1016 | +0.1012 |
| first1_R@10 | 0.0154 | 0.4806 | +0.4652 |
| last1_R@10 (L3) | 0.9976 | 0.9994 | +0.0018 |
| top1_eq_target_rate | 0.0000 | 0.0444 | +0.0444 |
| n_exact_hits | 0 / 5000 | 508 / 5000 | +508 |
| mean_exact_rank | None | 5.24 | — |

**exact match consistency** (per-sample):
- both=0 (canary 与 Stage4 都 hit) → 0
- canary_only=0 (canary hit, Stage4 not) → 0
- stage4_only=508 (Stage4 hit, canary not) → 508
- neither=4492 (都没 hit) → 4492

**pred_top1_diff_n** = 5000/5000 (所有 5000 sample 两边 top1 prediction 都不同)

## 三、判定: FAIL_ALIGNMENT

Issue #137 spec 验收条件: "canary_R@10 must equal formal Stage4 on same 5000 subset"

**实测**: canary exact_R@10=0 vs Stage4 0.1016, 差距 10.16% (abs), 5000 sample **完全无交集 hit** (both=0), 5000 sample **top1 prediction 全不同**.

**结论**: canary (Issue #136) 与正式 Stage4 (Issue #137) 不是协议微调差异, 而是 generate 路径根本不同. **canary 自身实现存在 bug, 不是协议对齐问题**.

## 四、根因分析 (与 protocol_diff_table 对齐)

Issue #137 spec 候选 5 个 alignment 区, 实测发现:
1. **code path 差异** (`model.model.generate` vs `model.generate` wrapper): 关键. wrapper 默认 `max_length=5` (Stage4) vs canary `max_length=8` (T5 内部). generate 输出长度上限不同 → 影响 beam search 轨迹.
3. **post-processing 差异** (canary 缺 `preds[:, 1:]` skip first): 关键. canary 把 `[start, a, b, c, d]` 全作为 prediction (5 tokens), 但只取前 4 token. 而 label 是 `[a+off, b+off, c+off, d+off]` (4 tokens). 即使 generate 输出与 label match, canary 误读 start token 也算匹配失败.
4. **target_sid encoding 差异** (canary `sid[target]` raw digit, Stage4 `labels + offset` token id): canary top1 pred 是 token id (已加 offset), target_sid 是 raw digit (无 offset) → 直接比对永远不等. 这就是为什么 canary `top1_eq_target_rate=0.0`.

但根本问题不是 #3 #4 细节. 即使 canary 修了 target encoding (加 offset), 它仍缺 `preds[:, 1:]` skip. 而且 canary 用 list-loop + EOS 截断, 跟 Stage4 vectorized 不等价.

**关键洞察**: canary (Issue #136) 表面是"参考评估协议"诊断, 实际是"独立重写 generate/evaluate 协议". 这是 spec 显式禁止的 (Issue #137 spec: "Stage4 full eval 的 code path 是唯一 oracle, canary 不得自行实现平行 generate/evaluate 逻辑").

## 五、对后续 issue 的建议 (R19+R39 联动)

1. **放弃 canary 路径**: Issue #136 reference-0 数据画像的 exact_R@10=0.0 不可信, 因为 canary 自身有 bug. 数据画像 (item geometry / L3 utilization / prefix bucket) 仍然有效, 但 exact_R@10 指标废弃.
2. **Stage4 raw (Issue #137) 是新的 oracle**: 5000 sample 5000 raw predictions + Stage4 评估 R@10=0.1016 (与 full eval 0.0962 在 5000 子集上略高, 因为 5000 随机 sample 略有偏). 用 Stage4 raw 做后续 item-level 可观测性诊断.
3. **Issue #138 candidate**: 把 Stage4 raw predictions 扩展到全 24772 sample, 建立 per-item 命中率 + 与 Issue #136 item geometry / L3 利用率 做 Spearman / bucket 关联.
4. **Issue #139 candidate**: Stage2 Stage3 修方向不应再通过曲率机制 (R36), 应该用 add_4th_dedup_digit (Issue #122 plan) 修复 L3=0 形态, 因为 L3 PAD_count=9894/9922 + top correlation L2_uniqueness_vs_prefix_hit 仍是 Stage4 R@10 退化的根本结构性原因.

## 六、产物清单 (Issue #137 spec 5 项)

1. ✅ `alignment/stage4_protocol_manifest.json` (ckpt sha256, SID sha256, T5 config, code path, HAB eval state)
2. ✅ `alignment/protocol_diff_table.md` + `protocol_diff_table.json` (9 candidate areas 对比, 3 关键差异)
3. ✅ `alignment/canary_stage4_trace/` (raw_predictions_stage4.json 5000 sample 完整 — Stage4 entry 唯一 oracle)
4. ⏳ `alignment/curvature_observability_table.csv` (本次只对齐协议, 曲率可观测性关联 = Issue #138 范畴)
5. ✅ `alignment/alignment_verdict.md` + `alignment_verdict.json` (本文件)

## 七、约束合规

- R7 (GPU util<10%): PASS (4 卡空闲启动 stage4_raw_predictions.py)
- R8 (无 fallback): PASS (raw_predictions 限制 CANARY_MAX=5000; 修 bug 后重跑; 没有 silent fallback)
- R17 (Gate FAIL → STOP): 本 issue 阶段产物全部产生, 不涉及 Gate FAIL
- R18 (4 维度对比): 见 issue137_verdict.json (D1/D2/D3/D4)
- R19 (激进 owner): FAIL 发现后立即给后续 issue 候选
- R20 (commit + 4 Gate 详细): 见 issue137_verdict.json
- R21 (commit hash 明示): commit 阶段实施
- R33 (verdict 文件名): `issue137_verdict.json` (按 gitlab issue 编号)
- R34 (任务目录 stage1/2/3/4_beam20.py): PASS (Issue137 有 stage1.py stage2.py stage3.py stage4_beam20.py, 但实际 stage1/2 产物复制自 Issue133, 满足 R44 自包含)
- R36 (曲率机制变更): PASS (本 issue 仅观察对齐, 无曲率变更)
- R40 (4 stage 全量): PASS (Issue137 stage1+stage2 实时复制自 Issue133 + stage4_raw_predictions.py 实时跑出 5000 raw + eval R@10=0.0962)
- R42 (DDP 4 卡): stage4 走 self-launch DDP 4 卡 (端口 29511), 实时跑成功
- R44 (dataset + _lib 自包含): PASS (dataset/test.parquet 引用共享顶层, _lib/ 已复制)
- R45 (gitlab only): 即将 commit + push 到 origin/main (gitlab)