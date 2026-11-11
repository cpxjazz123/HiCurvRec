# #179 R48κ2：asinh 无奇异形式已落地，**16/16 通过**；绝对路径清零

## TDD 的红 → 绿

1. **红**：把新测试 `tests/test_r48k2_coincident.py` 先跑在 #178 的 `arccosh` 实现上——`2 failed / 1 passed`，失败正是 `p=q` 与 `‖p−q‖=1e-12/1e-9/1e-6/1e-3` 四档下的 `nan` 梯度与 `gradcheck` 失败，复现了 #178 冒烟的死法。
2. **绿**：把 `distance` 换成 Issue 的 asinh 形式
   `d_κ = (2/√κ)·asinh(√κ·‖p−q‖_ε / √((1−κ‖p‖²)(1−κ‖q‖²)))`，
   `‖p−q‖_ε = √(‖p−q‖² + ε_pq)`，`ε_pq = finfo.eps · max(‖p‖²,‖q‖²,1)`；分母两因子以同一机器精度下限保护。改完 **16/16 通过**（含重合点梯度有限、`gradcheck` 通过、与 arccosh 形式在 x≥1e-3 等价、与冻结 `poincare_contract.distance` 在 κ=1 一致、`GEOMETRY_OFF` 与 `torch.cdist` 逐位相等、行为损失与 R48 逐位相等、梯度通路）。

重合点距离不再是精确 0，而是 `≈2√eps`（float64 下 3.1e-08），这是 ε 下限的必然结果，测试容差相应改为 `<1e-6` 并写明原因。

## 身份与路径（§19b0）

从 R48 继承的 `train_rqvae.py` 原本**硬编码绝对路径**（输出目录、embedding、train parquet、metrics、以及 `_LAUNCHER` 的 torchrun）。已全部改为：`ARM_NAME="TIGER_r48k2_rq"`、`ARM_VARIANT="R48K2"`、仓库根由标记目录（`stage2_RQ-VAE`/`results`/`stage1_GeneEmbedding`）向上推导，`RESULT_DIR = REPO_ROOT/"results"/"stage2_RQ-VAE"/ARM_NAME`，torchrun 由 `Path(sys.executable).parent` 推导。**树内 `/home/` 绝对路径计数 = 0**。

τ=0.07、权重调度→0.20、配对/过滤、Sinkhorn(ε=0.003、50 sweeps、centring)、码本 256、3000 epoch、seed 42、全局 batch 4096 全部保持 R48 原样。

## 冒烟 **PASS** → 正式训练已启动

`SMOKE_PASS`、`"passed": true`。`applied_grad_norms` 六个全为**有限正值**（0.0558–0.0973，无 NaN/Inf），#178 的 `[NaN]` 未再出现；冒烟 epoch 内 `collision=0.878879`（单 epoch 早期值，仅记录）。训练已起（6 个相关进程），首个记录 epoch 1455/3000、loss 2.99e-4。
