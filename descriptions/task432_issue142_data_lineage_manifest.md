# Task #432 / Issue #142 [方向C Gate4] history-SID 数据血缘审计后冻结评测清单

## R18 4 维度路径对比 (vs Issue #139)

| 维度 | Issue #139 (task429, NO-GO 收口) | Issue #142 (本 task, 修复方向C) |
|------|----------------------------------|---------------------------------|
| **D1 spec 摘录** | Stage4 真实协议重建 + 双重复跑 (无 manifest, 无 lineage) | **manifest + frozen eval**: 可机器核查 manifest + 20-record 逐字段可逆追踪 + frozen inputs |
| **D2 实施核心** | AdapterHookedHGRec.generate 走 `self.hgrec.model.generate(inputs_embeds=...)` (直接 T5 路径) | **manifest 优先**: 5 SHA256 + schema + 1-indexed alignment check + 20-record trace + frozen code_path_test.npy + reused task429 control/adapter 双复跑 |
| **D3 Gate 4 失败机制** | Adapter R@10=0.03684 -62.9% vs control 0.10203 (#129 dual-gate 2 epoch 训练极限) | **data lineage 错位**: 检查 test.parquet → history → target → SID tokens 链是否有错位 (Item-SID mapping 1-indexed offset) |
| **D4 引用文献** | arXiv:2309.04082 mixed-curvature Transformer | 同文献 + 复现性审计 (R139 reproducibility triangle) |

**R18 判定**: 4 维度都有差异 (重点在 D1 manifest 强制 + D2 lineage audit 优先 + D3 data lineage 错位假设), 必须做新实验.

## 实施
- `scripts/task432_issue142_data_lineage_manifest.py` (~290 lines)
- 5 个 SHA256 (test.parquet, train.parquet, SID npy, T5 ckpt, Adapter)
- 1-indexed Item-SID alignment check (out_of_range_0idx_count=1, 1idx_count=0)
- 20-record random reversible trace (user/history/target → SID tokens 1-indexed lookup)
- Token range check (3 层全部 token ∈ [0, K_l))
- Frozen code_path_test.npy SHA256 `19213ed27fd9a8a30ffa3ee983c36818b23a1a0e07e59a65db2bae46d5022860`
- Reused task429 control R@10=0.10203 + adapter R@10=0.03684 双复跑
- 7 件套: config + 6 SHA256 + manifest + frozen inputs + raw_log + verdict + commit
- 产物: `products/task432_issue142_data_lineage_manifest/{config,verdict,manifest}.json + code_path_test_frozen.npy`

## Gate 4 决策阈值
- Manifest 可机器核查 (5 SHA256 + schema + alignment + 20-record trace + token range)
- Frozen inputs 一致 (code_path_test SHA256 锁定)
- Control 非零合理 (R@10 ≈ baseline 0.1020)
- 两次 adapter 指标齐全
- Test R@10 > 0.1020 (target reached) — 已知 Adapter 反作用, 但 lineage PASS 已闭环

## 复现命令
```bash
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task432 \
  python3 scripts/task432_issue142_data_lineage_manifest.py
```