# Issue #144 — AF3 外部几何选择：候选清单与指标

冻结实现：`results/research/issue143/structural_report.py` 的 `structural_distortion`
（冻结 TRAIN 支持、共享校准、Huber），readout 按每个臂自己的导出几何重建：
`TIGER_RQ-VAE` 为 E32 全维求和；`TIGER_mixed_rq_E` 为 c=0 的 E16×E16；
F0/G/L/A 为 c=1 的 E16×H16 冻结 mixed readout。

## 候选与几何画像

| 臂 | L1 | L2 | L3 | 曲率 | α 规则 | SID/checkpoint |
|---|---|---|---|---|---|---|
| `TIGER_RQ-VAE` | E32 | E32 | E32 | — | 无 | 有 |
| `TIGER_mixed_rq_F0` | E16×H16 | E16×H16 | E16×H16 | c=1 | 固定 1/2 | 有 |
| `TIGER_mixed_rq_G` | E16×H16 | E16×H16 | E16×H16 | c=1 | 全局 α_ℓ（外层选择） | 有 |
| `TIGER_mixed_rq_L` | E16×H16 | E16×H16 | E16×H16 | c=1 | 按 prefix 的 α_ℓ(p) | 有 |
| `TIGER_mixed_rq_E` | E16×E16 | E16×E16 | E16×E16 | c=0 | 局部外层循环 | 有 |
| `TIGER_mixed_rq_A` | E16×H16 | E16×H16 | E16×H16 | c=1 | 固定 1/2 + AF3 辅助 | 有 |

全部六个候选的 `item_sids.json`、`rqvae_best.pth` 均在
`results/stage2_RQ-VAE/<arm>/`，SHA 记在 `geometry_selection.json` 的
`inventory` 与各候选的 `stage2_metrics/`。

## 统一指标（独立评估支持，seed=7，940498 对，真边 315391）

| 臂 | D_AF3 | ret@16 | Brier | min 码用量 | L3 碰撞 | 重建 MSE |
|---|---|---|---|---|---|---|
| `TIGER_RQ-VAE` | 0.2886 | 0.0495 | 0.2483 | 247 | 0.2745 | 6.67e-5 |
| `TIGER_mixed_rq_A` | 0.2861 | 0.0453 | 0.2459 | 50 | 0.4611 | 7.24e-5 |
| `TIGER_mixed_rq_E` | 0.2882 | 0.0467 | 0.2439 | 87 | 0.3354 | 7.14e-5 |
| `TIGER_mixed_rq_F0` | 0.2873 | 0.0462 | 0.2443 | 79 | 0.3891 | 7.10e-5 |
| `TIGER_mixed_rq_G` | 0.2848 | 0.0469 | 0.2456 | 64 | 0.4218 | 7.14e-5 |
| `TIGER_mixed_rq_L` | 0.2881 | 0.0465 | 0.2448 | 49 | 0.4410 | 7.29e-5 |

选择支持（seed=42）与评估支持（seed=7）的表逐项见
`geometry_selection.json` 的 `selection` / `evaluation`。

## 复算

```bash
/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
  results/research/issue144/select_geometry.py
```

输出 `results/research/issue144/geometry_selection.json`；不训练任何模型。
