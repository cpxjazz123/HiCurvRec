# Phase 0B: 防坍缩独立测试 (2026-07-27)

## 核心问题

利用率 23% 是不是一个可独立阻挡的设计信号? 用户决策:
> "塌缩是瓶颈. 0B: 利用率崩溃能不能被独立挡住 (1 小时, 1 张卡)"

## 判据

利用率从 ~23% 回到 ≥ 80% ✅ 死结解开, 进 Phase 1 / 仍然 < 50% ❌ 坍缩不是可独立解决的, **停止 Phase 1**.

## 设计

3 个独立试验, 同 D1-like 已知崩配置, **只换 --anti_collapse**:

| 试验 | --anti_collapse | 目的 |
|---|---|---|
| baseline | `none` | D1 对照 (复现坍缩) |
| + dead_revive | `dead_revive` | 用户第一推方案 (10 行, 简单兜底) |
| + simvq | `simvq` | SimVQ (Zhu et al.) W(B) 重参数化, 让死码字也持续收到梯度 |

## 已知崩配置 (D1-style)

| 参数 | 值 |
|---|---|
| Dataset | Musical_Instruments (R5) |
| Codebook | [64, 128, 256] |
| e_dim | 32 |
| Encoder layers | [4, 3, 2] |
| Product manifold | True, hyp_dim=4, euc_dim=28 |
| assignment_mode | per_codeword_kappa |
| c_k 更新 | spread (c_k_alpha=0.3, clamp [0.5, 2.0], dead_thr=20) |
| c_k 范围 | [0.5, 5.0] |
| Distance | rho_theta (稳定公式) |
| NormCap | **OFF** (诱发坍缩!) |
| beta | 0.5 |
| epochs | 50 |
| batch_size | 256 |
| lr | 1e-3 |
| loss_type | poincare |

## dead_revive 实施细则

- 每个 epoch 末 (`_vaild_epoch` 末尾) 累积全量 indices per-layer.
- 每层: `counts = bincount(indices)`, `dead = (counts == 0)`.
- 幸存 alive_idx = (counts > 0), 随机抽 n_dead 个码字 + 0.05 噪声 → 重 init `embeddings.weight.data[dead_mask]`.

> 用户原案: 噪声尺度 0.01-0.05, 阈值 counts==0 (最严).

## SimVQ 实施细则

- `codebook_base = Parameter(randn(K, e_dim), requires_grad=False)` 冻结随机基.
- `codebook_proj = Linear(e_dim, e_dim, bias=False)` 可学投影, init W ≈ 0.5 × I.
- 真 codebook = `codebook_proj(codebook_base)` (K, e_dim).
- 死码字通过 B 持续收到梯度 (跟 W 的链式), 不会真死.

> 用户原案: "B 冻结让梯度通过 W 流到所有码字, 死了的码字也持续收到梯度 → 不会完全归零或塌缩."

## 实施位置 (代码改动)

- `HG-Rec/train_hrqvae.py`:`--anti_collapse {none,dead_revive,simvq}` argparse + HRQVAE 构造透传.
- `HG-Rec/model/hrqvae.py`:`HRQVAE.__init__` 接收 + 透传到 HResidualVectorQuantization.
- `HG-Rec/model/utils.py`:
  - `HVectorQuantization.__init__` 接收 + 校验 + `self.anti_collapse`.
  - SimVQ: `codebook_base` / `codebook_proj` 注册; `get_codebook()` 用 W(B).
  - `HVectorQuantization.revive_dead_codes(indices, latent, dead_thr, perturb)`: 重 init 死码字.
- `HG-Rec/model/hrqvae_trainer.py`:`_vaild_epoch` 末尾累积全量 indices per-layer, 对每层调用 `revive_dead_codes`.
- `scripts/phase0b_test.sh`:3 个 launcher, 同 D1-like, 只换 `--anti_collapse`.

## 判定

| 结果 | 后续 |
|---|---|
| **三个里至少一个** util ≥ 80% | 进 Phase 1 (4 臂 H0/H1/H2/H3) |
| **三个都** util < 50% | **停止 Phase 1**, 转写作 |

## 进度

- 22:54 — Phase 0B orchestrator 启动 (PID 1538615), GPU 0
- 待: anti_none 50 epoch → anti_dead_revive 50 epoch → anti_simvq 50 epoch
