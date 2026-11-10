# #156 派发记录

- 门 5（相对）、门 6（全局）重测通过，未改任何常数。
- 首次派发在**真实入口一 epoch 冒烟**处失败：`Expected all tensors to be on the same device` —— 冻结目标与 `d_ref` 在 CPU、前缀在 GPU。修复：`struct_loss` / `triangle_term` 把 `target`、`d_ref` 移到距离张量的设备与 dtype，`edge_distance` 把边索引移到前缀设备，空张量返回值改为继承设备。回归测试 `tests/test_device_placement.py`（CPU 目标 + 前缀几何必须得到同设备标量）。
- 修复 commit `83bdbb86`；重派发后冒烟 **SMOKE_PASS**：24 步、`applied_grad_norms` 8 个全为正（51.97 → 0.46）、`dead_terms=[]`、`nonfinite_terms=[]`、导出 SID `(24587, 4)` 唯一 24587。
- 冒烟期 collision 0.999959（1 epoch、未训练），仅记录。
- 正式训练：seed 42、3000 epoch、全局 batch 4096、四卡 50202，逐 epoch 在 `.cursor/hypothesis.md` 之外只写 `logs/training_metrics.jsonl`；产物目录 `results/stage2_RQ-VAE/TIGER_af3_target_distance_rq`。
