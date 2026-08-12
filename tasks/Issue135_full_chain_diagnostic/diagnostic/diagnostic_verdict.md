# Issue #135 全链路纯诊断 verdict

## 参考 vs 当前

| 项 | reference (Task #84 baseline) | current (Issue #133) | 状态 |
|---|---|---|---|
| git commit | unavailable | `76b1155cb39a6e4437db720e90c760541bd1cf01` (Issue #133 commit) | unavailable |
| SID SHA256 | unavailable | `d43dea6fe1988105ec06f8035e62028ad2d3e67a54a6fca0592f0bc004a84158` | unavailable |
| Stage1 item_emb SHA256 | unavailable | 见 provenance_manifest.json | unavailable |
| Stage3 best valid_R10 | 0.1267 (CLAUDE.md) | 0.1251 | **different** (-1.3%) |
| Stage4 test_R@10 | 0.1024 (CLAUDE.md) | 0.0962 | **different** (-6.05%) |

## 最早可证实分叉点

由于 reference run (Task #84 baseline) 的 ckpt / code / log 在本仓库不可用 (unavailable),
无法直接对比 reference vs current 的端到端 provenance。

**已知最早 observable 差异**: Stage3 best_valid_R10=0.1251 < reference 0.1267 (-1.3%).
但这已是训练后的 metrics, 无法定位"哪个 stage 最早分叉"。

## 推测 (但无 reference 可证)

- Stage1: item_emb.parquet 47MB max_norm=0.6603, 但不知道 baseline max_norm
- Stage2: SID 4-digit L3=1 (全 PAD, 这是 v15 capmatch dedup digit 应有的形态)
- Stage3: best_valid_R10=0.1251 (-1.3% vs baseline 0.1267), 训练行为可能与 baseline 不一致
- Stage4: test_R@10=0.0962 (-6.05% vs baseline 0.1024), eval 可能与 baseline 不一致

## 后续建议

1. 寻找 Task #84 baseline commit hash (可能在 gitlab 其他 branch)
2. 比对 baseline v22b Stage3 vs 本仓库 Stage3 实现, 找出具体差异
3. 可能方向:
   - Stage3 LR / scheduler / early_stop 实现
   - Stage3 label_smoothing / weight_decay
   - Stage3 DDP 同步 / 数据 shuffling
   - Stage4 beam_search 算法 (length_penalty, repetition_penalty 等)

## 完成判据

- [x] A 全局可复现性 + 环境采集完成 (current run 完整, reference unavailable)
- [x] B 数据与划分采集完成 (SHA / shape / dtype)
- [x] C Stage1/2 产物链采集完成 (hash + norm + SID utilization)
- [x] D Stage3 训练行为采集完成 (params + log + metrics)
- [x] E Stage4 解码与评估采集完成 (beam config + eval metrics)
- [x] 诊断矩阵产物 (manifest, csv, verdict)
- [ ] reference vs current 逐样本 prediction_diff.csv (缺 reference ckpt, unavailable)
- [ ] Stage4 100-item canary trace (缺 reference ckpt, unavailable)

## 结论

诊断矩阵已建立, 但因 reference (Task #84 baseline) ckpt / code 在本仓库 unavailable,
无法证实"最早分叉点"在哪个 stage。已知最早 observable 差异是 Stage3 best_valid_R10
下降 1.3%, 提示 Stage3 训练行为本身可能与 baseline 不一致, 而非 Stage2 量化差异。
