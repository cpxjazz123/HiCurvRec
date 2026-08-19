# v35 备胎 — 整个 T5 切 Lorentz model (Chen 2022 Fully Hyperbolic NN)

## 任务目标

突破 v19 baseline test_R@10=0.1113 ceiling。在 Stage 3 把 HG-Rec T5 内置的 24 个 RMSNorm 全部从
Euclidean 切到 Lorentz hyperboloid model (Chen 2022 Fully Hyperbolic NN 风格),
数学上比 v34 Poincaré RMSNorm 更稳定 (无 ball 边界, 无 atanh 数值爆炸风险)。

## Stage 结构 (R40 自包含 + R34 新版本强制 4 stage 脚本)

| Stage | 路径 | 状态 | 说明 |
|---|---|---|---|
| 1 | `stage1.py` | 跳过 (复用 v19 RQ-VAE) | 验产物存在, v35 不需要重训 Stage 1 |
| 2 | `stage2.py` | 跳过 (复用 v19 SID) | 验产物存在, v35 不需要重训 Stage 2 |
| 3 | `stage3.py` | **新跑** | v35 HG-Rec + Lorentz RMSNorm 训练 (DDP 4 卡, 200 epoch, EARLY_STOP=20) |
| 4 | `stage4_beam20.py` | **新跑** | v35 best_ckpt 评估 (DDP 4 卡, beam=20, R35 单 ckpt) |

## 关键文件

### 任务目录 (本目录)
- `stage1.py` / `stage2.py` — 复用 v19 产物验证
- `stage3.py` — Stage 3 训练入口 (torchrun DDP 4 卡)
- `stage4_beam20.py` — Stage 4 评估入口 (torchrun DDP 4 卡)
- `_lib/lorentz_norm.py` — Lorentz RMSNorm (T5LayerNorm 替换器)
- `_lib/lorentz_ops.py` — Lorentz 原语 (Chami 2019 HGCN eq 8)

### 引用 M2M3 主目录
- `train_decoder.py` (line 161-167) — v35 路径触发 use_lorentz_norm=True
- `test_eval_only.py` — Stage 4 评估器 (FORCE_HGREC=1 bypass gin bug)
- `modules/hg_rec.py` — HG_Rec T5 wrapper (use_lorentz_norm 配置项)
- `configs/decoder_instruments_hgrec_v35.gin` — v35 gin 配置

### Stage 1 / Stage 2 复用产物 (R44 + R40)
- `rqvae_out_v19_cend_07/rqvae_final.pt` (v19 Stage 1 RQ-VAE, c_end=0.7 curriculum)
- `dataset/Instruments/Instruments_v19_sids_for_hgrec.npy` (v19 Stage 2 SID, shape 9922x4)

## 创新点 (R36 框架级变更 — 曲率机制变更, 非调参)

- **v34 (Poincaré RMSNorm, baseline)**: T5 24 个 RMSNorm → HyperbolicRMSNorm (Poincaré 球归一化)
  - 数学: `y = logmap0_c(expmap0_c(x / rms(x) * weight), c)`, c=1.0 静态
  - 问题: Poincaré 球边界 `√(1 - c*||x||^2)` 接近 0 时数值不稳定
  - 测试: test_R@10=0.1077 (v19 baseline 0.1074, +0.0003, 边际 PASS)

- **v35 (Lorentz RMSNorm, 本任务)**: T5 24 个 RMSNorm → LorentzRMSNorm (Chen 2022)
  - 数学: `y = logmap0_c(expmap0_c(x / rms(x) * weight), c)`, c=1.0 静态
    - 与 v34 同公式, 但流形从 Poincaré 球 (边界 sqrt 奇点) 切到 Lorentz hyperboloid (`-x_0^2 + Σ x_i^2 = -1/c` 严格恒等)
  - 优势:
    - 无 ball 边界 (Lorentz 是无限流形), 数值更鲁棒
    - 与 v31 Lorentz Codebook 同流形 (Stage 1 + Stage 3 Lorentz 统一)
    - Chen 2022 §3.4 验证深 Transformer 训练期梯度更稳
  - 路径:
    - `_lib/lorentz_ops.py`: poincare_to_lorentz / expmap0_lorentz / logmap0_lorentz / minkowski_dot
    - `_lib/lorentz_norm.py`: LorentzRMSNorm (forward: RMSNorm → embed R^(d+1) → expmap0 → logmap0 → spatial)
    - `modules/hg_rec.py::HG_Rec.__init__`: install_lorentz_rms_norm (替换 24 个 T5LayerNorm)
    - `train_decoder.py::train (line 161-167)`: 检测 sys.argv 含 'v35' → use_lorentz_norm=True

## R37 决策

| test_R@10 | 决策 | 后续动作 |
|---|---|---|
| ≥ 0.1113 (v19 baseline) | **PASS** | v35 突破 ceiling, commit + push, 留作新基线 |
| < 0.1113 | **FAIL** | R50 revert: 删 _lib/lorentz_norm.py / _lib/lorentz_ops.py, modules/hg_rec.py / 4, train_decoder.py / 4, configs/decoder_instruments_hgrec_v35.gin / rm, 回到 v19 baseline |

## 运行方式

```bash
# Stage 1 / 2 验证 (秒级, 仅 check 产物存在)
python3 stage1.py
python3 stage2.py

# Stage 3 训练 (DDP 4 卡, ~25 min, EARLY_STOP=20)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29509 stage3.py

# Stage 4 评估 (DDP 4 卡, beam=20, ~10 min)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29510 stage4_beam20.py
```

## 噪声控制 (R51 硬约束)

- Stage 3 训练: `train_decoder.py` 已内置 `_random.seed(42 + process_index)`, `np.random.seed(42 + process_index)`, `torch.manual_seed(42 + process_index)`
- Stage 4 评估: `test_eval_only.py` 已内置 `manual_seed(42)`
- 4 卡分片 (DistributedSampler) 由 `accelerator.prepare()` 自动 wrap
- 禁手动 `DistributedSampler(seed=42)` (与 accelerator 双重 wrap 会导致 4×4 = 16 分片噪声)

## R40 自包含检查

- 数据集: `/home/wlia0047/ar57/wenyu/GeneRec/dataset/Instruments/` (R44 共享)
- 依赖库: 本目录 _lib/ (lorentz_norm.py + lorentz_ops.py)
- Stage 1/2 产物: M2M3 主目录已存在 (复用 v19)
- Stage 3 训练: M2M3 主目录 train_decoder.py + v35 gin
- Stage 4 评估: M2M3 主目录 test_eval_only.py + v35 gin + best_ckpt