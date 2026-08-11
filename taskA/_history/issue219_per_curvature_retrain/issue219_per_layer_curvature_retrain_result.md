# Issue #119 / #219 — Per-layer Curvature 重训验证结果

日期: 2026-08-11
方法: 8 κ × 100 epoch Stage2 重训 (DDP 4-card, 复用 Issue #210 equal128 Stage1)
扩展: CURV_FIXED_KAPPAS = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0] (issue 推荐方式 A)

## 1. Stage2 ckpt 路径 (8 κ)

- κ=0.01 → `/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/issue219_per_curvature_retrain/c_fixed_k0p01/hrqvae_kappa_sync.ckpt`
- κ=0.1 → `/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/issue219_per_curvature_retrain/c_fixed_k0p1/hrqvae_kappa_sync.ckpt`
- κ=0.5 → `/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/issue219_per_curvature_retrain/c_fixed_k0p5/hrqvae_kappa_sync.ckpt`
- κ=1.0 → `/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/issue219_per_curvature_retrain/c_fixed_k1/hrqvae_kappa_sync.ckpt`
- κ=2.0 → `/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/issue219_per_curvature_retrain/c_fixed_k2/hrqvae_kappa_sync.ckpt`
- κ=5.0 → `/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/issue219_per_curvature_retrain/c_fixed_k5/hrqvae_kappa_sync.ckpt`
- κ=10.0 → `/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/issue219_per_curvature_retrain/c_fixed_k10/hrqvae_kappa_sync.ckpt`
- κ=20.0 → `/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/issue219_per_curvature_retrain/c_fixed_k20/hrqvae_kappa_sync.ckpt`

## 2. Pair-wise Distortion (self-curvature)

| κ | L0 | L1 | L2 |
|---|---|---|---|
| 0.01 | 1.012604570860276e-06 | 6.085560499968778e-08 | 1.618987610640943e-08 |
| 0.1 | 1.3351354937185533e-06 | 5.2030202368769096e-08 | 1.3934689846450965e-08 |
| 0.5 | 1.162222270068014e-05 | 4.479381843225383e-08 | 1.0271279826667978e-08 |
| 1.0 | 4.9202011723537e-05 | 1.0497756619542997e-07 | 3.4443946361761846e-08 |
| 2.0 | 9.084250632440671e-05 | 3.4140916227443086e-07 | 1.072491073728088e-07 |
| 5.0 | 0.0005128264892846346 | 2.984325192301185e-06 | 1.053330379363615e-06 |
| 10.0 | 0.0009252488380298018 | 1.4305713193607517e-05 | 2.8340368771750946e-06 |
| 20.0 | 0.00135247060097754 | 2.8527210815809667e-05 | 9.228676390193868e-06 |

## 3. 跨曲率 sweep (Cross-C Distortion)

详见 `curvature_retrain_distortion_cross.csv`. 9 点 c∈[0, 10] 同 Issue #83 网格.

## 4. Codebook 健康度

| κ | util | dead_ratio |
|---|---|---|
| 0.01 | 0.08252091705799103 | None |
| 0.1 | 0.08245113740364711 | None |
| 0.5 | 0.0809013769030571 | None |
| 1.0 | 0.08211549123128255 | None |
| 2.0 | 0.07338418439030647 | None |
| 5.0 | 0.06618958339095116 | None |
| 10.0 | 0.05564612460633119 | None |
| 20.0 | 0.047645825892686844 | None |
