---
type: verdict
created: 2026-08-02
tags:
  - paper
up: "[[index]]"
---
# Paper Table 6 默认值对齐 v2 (2026-07-25 用户跟进)

> **核心改动**: 按用户指令"按照你找到的不同的配置，进行修改对齐论文", 在 Phase 0.2 已修过的基础上**第二轮对齐**, 改 4 处 `train_hrqvae.py` / `train_HG-Rec.py` defaults. 已 py_compile 通过.
>
> **重要警告**: 这次改动只改 defaults, **未启动任何训练**. 之前 task178/task180 (Phase 0.2 第一轮已修过) 已暴露 85.24% / 81.70% RQ-VAE mode collapse. 新 defaults 要等用户决策是否启动新 Stage 1 才能验证.

---

## 1. 4 处修改一览

| # | 文件 | 参数 | 旧 default | 新 default | paper 来源 |
|---|------|------|-----------|-----------|-----------|
| 1 | `HG-Rec/train_hrqvae.py:17` | `--epochs` | 200 | **1000** | Table 6 Instruments "1000 epoch" |
| 2 | `HG-Rec/train_hrqvae.py:18` | `--batch_size` | 1024 | **256** | Table 6 (跟 GR 训练 batch 一致) |
| 3 | `HG-Rec/train_hrqvae.py:33` | `--sk_epsilons` | `[0.003, 0.003, 0.003]` | **`[0.0, 0.0, 0.0]`** | paper Eq (8) 走 argmin 路径, 不引入 Sinkhorn (Sinkhorn 是 LETTER/TIGER 那条线) |
| 4 | `HG-Rec/train_HG-Rec.py:121` | `--num_layers` | 6 | **4** | Table 6 GR config "num_layer 4" (encoder 4 + decoder 4, 已对齐) |

**未改的 paper 字段** (默认已对齐或 launcher 已显式传):
- `lr=1e-3` (Table 6): ✅ 默认已对齐
- `weight_decay=0` (Table 6): ✅ 默认已对齐
- `learner=AdamW` (Table 6): ✅ 默认已对齐
- `beta=0.5` (Table 6 Instruments Appendix G.2 peak): ✅ Phase 0.2 已修
- `c=1.0` (Table 6): ✅ `--loss_type poincare` 默认 + κ hard-coded 1.0
- `num_emb_list=[64,128,256]` (Table 6): ✅ Phase 0.2 已修
- `layers=[512,256,128,64]` (Table 6): ✅ 默认已对齐
- GR `num_decoder_layers=4`: ✅ 默认已对齐
- GR `d_model=128`, `d_ff=1024`, `num_heads=6`, `d_kv=64`: ✅ 默认已对齐
- GR `batch_size=256`, `lr=1e-4`, `dropout_rate=0.1`, `vocab_size=1025`, `early_stop=20`, `beam_size=20`: ✅ 默认已对齐

---

## 2. paper Table 6 (Instruments) 完整对照表

| 字段 | paper 值 | 旧 default (上游开源版) | 新 default (本次对齐) | 状态 |
|------|----------|------------------------|----------------------|------|
| **HRQ-VAE** | | | | |
| epoch | 1000 | 200 | **1000** | ✅ |
| lr | 1e-3 | 1e-3 | 1e-3 | ✅ |
| weight_decay | 0 | 0 | 0 | ✅ |
| learner | AdamW | AdamW | AdamW | ✅ |
| β (Instruments peak) | 0.5 | 0.5 (Phase 0.2 修) | 0.5 | ✅ |
| c (curvature) | 1.0 | 1.0 | 1.0 | ✅ |
| codebook size | [64,128,256] | [64,128,256] (Phase 0.2 修) | [64,128,256] | ✅ |
| layers (encoder) | [512,256,128,64] | [512,256,128,64] | [512,256,128,64] | ✅ |
| **batch_size** | 256 | 1024 | **256** | ✅ |
| **Sinkhorn** | OFF (argmin) | ON (0.003) | **OFF (0.0)** | ✅ |
| **GR (T5-mini)** | | | | |
| num_layer | 4 | 6 | **4** | ✅ |
| num_decoder_layers | 4 | 4 | 4 | ✅ |
| d_model | 128 | 128 | 128 | ✅ |
| d_ff | 1024 | 1024 | 1024 | ✅ |
| num_heads | 6 | 6 | 6 | ✅ |
| d_kv | 64 | 64 | 64 | ✅ |
| batch_size | 256 | 256 | 256 | ✅ |
| lr | 1e-4 | 1e-4 | 1e-4 | ✅ |
| dropout | 0.1 | 0.1 | 0.1 | ✅ |
| vocab_size | 1025 | 1025 | 1025 | ✅ |
| early_stop | 20 | 20 | 20 | ✅ |
| beam_size | 20 | 20 | 20 | ✅ |
| GR epoch | 200 | 200 | 200 | ✅ (paper 没说, 沿用上游) |
| max_len | 20 | 20 | 20 | ✅ (paper 没说, 沿用上游) |

---

## 3. 关键决策点 (R11.3 自主决策 + 备选)

### (a) Sinkhorn ON 还是 OFF
- **paper Eq (8)**: 只写了 VQ commitment loss, 没提 Sinkhorn. Sinkhorn 是 LETTER/TIGER 那条线 (RCQ 借鉴), 不是 HG-Rec 原生组件.
- **决定**: OFF (`sk_epsilons=[0.0, 0.0, 0.0]`)
- **备选**: ON (跟 phonism 一致, 强制 balanced assignment). 不选是因为这不属于 paper recipe.
- **风险**: OFF 后 Stage 1 codebook utilization 可能掉, 但 paper Table 7 Instruments 既然能 R@10=0.1315, 说明 paper argmin 路径够用.

### (b) batch_size 256 vs 1024
- **paper Table 6 (Instruments)**: batch_size=256 (跟 GR 训练 batch 一致).
- **决定**: 256.
- **备选**: 1024 (上游开源版默认, 单 epoch 快, 但更新次数减半). 不选是因为跟 paper 不一致.
- **风险**: batch_size 256 + 1000 epoch + 9922 items → 单 epoch 39 个 update × 1000 = 39000 total updates. 比 1024 batch 的 10 update/epoch × 1000 = 10000 updates 多 3.9×. 训练时间大约 ×3.9 (但收敛更稳).

### (c) num_layers 4 vs 6 (GR)
- **paper GR config**: num_layer=4 (encoder 4 + decoder 4 = 8 层, 比 t5-small 的 6+6=12 层浅).
- **决定**: 4.
- **备选**: 6 (上游开源版 t5-small 默认). 不选是因为 paper 自己用 4.
- **风险**: 模型参数量下降 (T5-small 6+4 配置约 5.5M, 4+4 配置估计 ~4M). 容量更小可能限制 T5 学习能力.

### (d) epochs 1000 vs 200
- **paper Table 6 (Instruments)**: 1000 epoch.
- **决定**: 1000.
- **备选**: 200 (上游开源版 default, 跑得快). 不选是因为跟 paper 不一致.
- **风险**: 1000 epoch × batch_size 256 = 单次 Stage 1 训练时间大约 5× task178 (200 ep × batch 1024 = ~30 min → ~2.5 h 1000 ep × batch 256).

---

## 4. 未启训练的原因 (R7 + R11.3)

1. **R7 GPU 检查**: 当前 GPU 0/1/2 还被 #178/#179/#180 Stage 3 占用, **新 Stage 1 启动会抢 GPU**, 违反 R7. 等用户决策 #179/#180 kill/归档后再启动.
2. **R11.4 不可逆决策**: 启新 Stage 1 = 投入 ~2.5 h GPU 时间. Phase 0 fix 自己引入 mode collapse 的根因 (Poincaré boundary saturation + 200 epoch 长训) **还没正式修复** (用户 2026-07-25 verdict §4 Option C 提议 commitment MSE + norm 正则化). 用新 defaults 启训前, 最好先决定是否叠加 norm 正则化, 否则只是"换一个 default 继续崩".
3. **user 已说"先别叠"**: 2026-07-25 verdict §3 明确 — 在崩的基础上叠结构是无效对照. 先修地基, 再叠.

---

## 5. 验证清单

- ✅ `python3 -m py_compile HG-Rec/train_hrqvae.py HG-Rec/train_HG-Rec.py` 通过
- ⏸ 单元测试 (建议): 验证 `sk_epsilons=[0.0, 0.0, 0.0]` 时 `forward` 走 argmin 分支不调 Sinkhorn. (脚本可从 task178_quantizer_unit_test.py 继承, 未在本轮新建.)
- ⏸ Stage 1+2+3+4 端到端: 等用户决策后启动.

---

## 6. 产物路径

| 类型 | 路径 |
|------|------|
| 改动文件 | `HG-Rec/train_hrqvae.py:17,18,33` |
| 改动文件 | `HG-Rec/train_HG-Rec.py:121` |
| 本 verdict | `verdicts/paper_aligned_defaults_v2_verdict.md` |
| 上轮 verdict | `verdicts/phase0_mode_collapse_verdict.md` (2026-07-25 第一轮, mode collapse 发现) |
| 关联 memory | `memory/phase0-mode-collapse.md` |

---

## 7. 用户待决策 (本 verdict 范围)

| 选项 | 内容 | 推荐 |
|------|------|------|
| A | **立即用新 defaults 启训** (Phase 1 复跑, paper 1000 epoch + batch 256 + Sinkhorn OFF + GR 4 层) | ❌ 不推荐 — mode collapse 根因没修, 新启只是换个 default 继续崩 |
| B | **先修 Phase 0 mode collapse 根因** (commitment MSE + ‖x‖² ≤ α 正则化), 再用新 defaults 启训 | ✅ 推荐 — 先把地基修干净, 否则 paper 对齐数字不可解读 |
| C | **继续等 #179/#180 自然跑完**, 期间在 CPU 上做 Phase 0 修复 unit test + ckpt 健康度检测 (50 epoch ≤ collision ≤ 30%) | ✅ + 推荐 — 不抢 GPU, 同时推进修复 |

**建议回复**: "C" 或 "C + Phase 0 修复 (commitment MSE)". 启动新训练前必须先有 unit test + norm 正则化 commit.

当前任务已完成，请做下一个任务的指示。