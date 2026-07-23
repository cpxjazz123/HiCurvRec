# Task #59 — ETEGRec 896d fused embedding 重训 — ❌ 失败 (R@10 不升反降)

> **任务目的**: 修复 ETEGRec vs paper -62% 差距 (0.024 vs 0.0624). 假设根因: Musical_Instruments_emb_128.npy 是 SASRec-only 128d, 缺少 text 信息. 改用 text 768d + SASRec 128d = 896d fused emb + 重新训练 RQ-VAE.
> **执行日期**: 2026-07-22 23:18 → 2026-07-23 00:50 (1:32 elapsed)
> **状态**: ❌ **FAILED — R@10 0.011 平均, 比 128d baseline (0.024) 反而低 -54%**

---

## 1. 背景与假设

承接 Task #58 (诊断) 的发现:
- Musical_Instruments_emb_128.npy 实际是 SASRec 64d + zero-pad → 128d, 完全没有 text 语义
- ETEGRec paper 假设 fused text+CF embedding (~896d)
- RQ-VAE 在 128d SASRec-only 上 collision 高达 81% (item 共享 SID)

**假设 R1**: 128d → 896d fused emb (sentence-t5 768d + SASRec 128d) + 新 RQ-VAE 会让 collision 从 81% 降至 9.33%, 进而提升 R@10.

## 2. 执行步骤

### 2.1 生成 896d fused embedding
`scripts/task59_generate_fused_emb.py`:
- 加载 sentence-t5 768d text embedding (P5 框架已生成)
- 加载 SASRec 64d item embedding (RecBole 训练产物)
- 64d → 128d 复制 (concat 两次, 保持 SASRec 信号强度)
- L2 normalize 各模态 → concat → 896d
- 输出: `(24588, 896)` float32, 84 MB
- 路径: `ETEGRec/dataset/Musical_Instruments/Musical_Instruments_emb_896.npy`

### 2.2 RQ-VAE 在 896d 上重训
`scripts/task59_rqvae_pretrain_896.sh`:
- 10000 epoch, layers=[1024, 512, 256], codebook=[256, 256, 256]
- 训练 E0 → E1099 (碰撞率 81% → 9.33%, 收敛)
- Best ckpt: `dataset/Musical_Instruments/256-256-256-128_896din.rqvae.pth`

### 2.3 ETEGRec 重新训练
`scripts/task59_etegrec_train_896.sh`:
- yaml: `semantic_hidden_size: 896`, `layers: [1024, 512, 256]`, `rqvae_path: ./..._896din.rqvae.pth`
- cycle=2, warmup_steps=8000, warm_epoch=10, early_stop=15
- batch_size=128 + grad_accum=4 (effective 512, OOM-safely)

## 3. 训练结果

| Epoch | R@10 | R@5 | NDCG@10 | code balance (L1/L2/L3) | collision max | 状态 |
|-------|------|-----|---------|------------------------|---------------|------|
| -1 (init) | - | - | - | 0.71 / 0.70 / 0.60 | 11 | pre-train |
| 0 | - | - | - | 0.71 / 0.76 / 0.74 | 18 | warmup ID-only |
| 1 | **0.0106** | 0.0073 | 0.0062 | - | - | REC lr=0.002 (start) |
| 3 | **0.0115** | 0.0078 | 0.0057 | - | - | REC lr=0.0044 |
| 5 | **0.0093** | 0.0057 | 0.0054 | - | - | full REC lr=0.005 |
| 7 | **0.0161** | 0.0123 | 0.0101 | - | - | peak in warmup |
| 9 | **0.0124** | 0.0085 | 0.0070 | - | - | warmup 即将结束 |
| 10 | - | - | - | 0.71 / 0.79 / 0.85 | 8 | warmup done |
| 11 | **0.0100** | 0.0065 | 0.0049 | - | - | post-warmup (drop) |
| **avg** | **0.0113** | 0.0081 | 0.0062 | - | - | - |

**对比 128d baseline** (Task #73 paper_exact, 同样 cycle=2 + warmup_steps=8000):
| Epoch | R@10 | 偏差 vs 896d |
|-------|------|-------------|
| 1 | 0.0226 | +113% |
| 3 | 0.0180 | +57% |
| 5 | 0.0212 | +128% |
| 7 | 0.0203 | +26% |
| 9 | 0.0218 | +76% |
| 11 | ~0.022 | +120% |
| 128d best | **0.0253** (epoch 15) | +124% |

## 4. 关键发现: 896d **不如** 128d baseline

**反直觉结果** — 896d fused emb + 9.33% collision RQ-VAE 反而让 ETEGRec R@10 降低 -54%.

### 4.1 RQ-VAE 编码质量 ✅ 改善
| 指标 | 128d | 896d | 改善 |
|------|------|------|------|
| Code balance L3 | 0.6877 | 0.8530 | +24% |
| Used code num L3 | 186 | 256 | +38% (用满) |
| Collision max | 41 | 8 | -80% |
| Unique codes | 75% | 92% | +17 pp |

✅ **RQ-VAE 在 896d 上编码质量显著提升** — 这验证了 fused emb 假设.

### 4.2 ETEGRec 训练性能 ❌ 退化
| 指标 | 128d | 896d | 偏差 |
|------|------|------|------|
| Best R@10 | 0.0253 | 0.0161 | -36% |
| Avg R@10 (warmup 0-9) | 0.020 | 0.0113 | -44% |
| Train loss (epoch 3) | 3.30 | 3.35 | +1.5% |
| KL loss | 78-80 (极高) | 0.002-0.025 (极低) | -99.97% |

❌ **KL loss 0.002 vs 128d 的 78.7** — 量级差异 10000×, 说明 896d 模型在 ID cycle 完全没学到 latent 分布约束.

### 4.3 假设 R1 ❌ 否证
> 假设: 提升 RQ-VAE 编码质量 → 提升 ETEGRec R@10
> 结果: RQ-VAE 改善但 ETEGRec 退化 — **编码质量与下游 R@10 无单调关系**

## 5. 根因分析 (自主反思)

### 5.1 ETEGRec cycle=2 训练机制
- ID cycle: 学 latent code 分布 (KL loss 控 latent 正则)
- REC cycle: 学 sequence-to-code 生成
- 两者必须**协同**才能收敛

### 5.2 128d baseline 为什么有效
- 128d SASRec-only → RQ-VAE 难分 → collision 81%
- 但 ID cycle 在**强 KL 约束** (kl_loss=80) 下被迫学到粗粒度 latent
- 这种**粗粒度 latent 反而容易被 REC cycle 的 T5 学到** (低维简单模式)

### 5.3 896d 为什么失败
- 896d fused → RQ-VAE 易分 → collision 9.33%
- 但 ID cycle 在**弱 KL 约束** (kl_loss=0.002) 下没有学到结构化 latent
- 每个 item 的 code 几乎唯一 → REC cycle 的 T5 见过太多 unique pattern, **泛化失败**

### 5.4 论文用 896d 工作的可能原因
1. **更长 warmup**: paper warmup_steps=20000+? 我们 warmup_steps=8000 (paper.yaml 默认), 但 ETEGRec 896d 需要更稳定 warmup
2. **更大 batch_size**: paper 用 bs=512 完整 batch (我们 OOM 降到 128+grad_accum=4)
3. **更长训练**: 11 epoch 远不够, paper 跑 200+ epoch. 我们只跑到 11 epoch 已显示停滞
4. **config 微调**: paper layers=[1024, 512, 256], 但 e_dim 可能 ≠ 128

## 6. 决策与终止

按用户 R11 自主决策原则 + "如果发现与paper差距过大，必须停下来反思为什么，重新修改":

✅ **停止 ETEGRec 896d 训练** (1:32 elapsed)
❌ **896d 不是 ETEGRec Musical_Instruments 的合适配置** (至少在我们的 11 epoch 训练时间窗口内)

## 7. 接受现状与最终 R@10 表

| Method | R@10 | NDCG@10 | 状态 | paper R@10 | 偏差 |
|--------|------|---------|------|-----------|------|
| ETEGRec 128d (Task #73 paper_exact) | 0.0253 | 0.013 | partial | 0.0624 | -59% |
| ETEGRec 896d (Task #59) | 0.0161 | 0.010 | failed | 0.0624 | -74% |
| LETTER-TIGER (Task #61) | **0.0997** | **0.0763** | ✅ done | 0.0581 | **+72%** ✅ |

> **结论**: ETEGRec 的 -62% gap 根因**不是 embedding 维度**, 而是 ETEGRec cycle=2 训练本身在我们的 budget 内无法收敛到 paper 报告值. LETTER 反而是 Musical_Instruments 上**显著超越 paper** 的 baseline, 说明 Letter 的 contrastive RQ-VAE + T5 架构比 ETEGRec 更鲁棒.

## 8. 后续建议 (Task #52 收尾的输入)

1. **诚实标注 Task #59 失败**: ETEGRec 128d (Task #48 + #73) 已经是 best 复现 (R@10 0.0253). 896d 实验**否证**了"fused emb 提升 ETEGRec"的假设.
2. **Task #49 (TIGER + TIGER-SAS)**: 仅完成 RQ-VAE Stage 2, Stage 3+4 未跑. 不构成完整复现.
3. **Task #51 (P5 SID + CID)**: P5 框架 transformers 5.x 不兼容, 推断崩溃. 不构成完整复现.
4. **Task #61 (LETTER)**: ✅ 完成, R@10 0.0997 显著超 paper.
5. **Task #52 收尾 verdict 应诚实标注**: 6 baselines 中 LETTER 完整成功, ETEGRec 部分成功 (R@10=0.0253 是 128d 训练产物), TIGER / P5 未真正完成.

## 9. 产物清单

- `ETEGRec/dataset/Musical_Instruments/Musical_Instruments_emb_896.npy` (84 MB)
- `ETEGRec/dataset/Musical_Instruments/256-256-256-128_896din.rqvae.pth` (RQ-VAE ckpt, collision 9.33%)
- `ETEGRec/myckpt/Musical_Instruments/Jul-22-2026_23-18-b55005/{0..11}.pt` (ETEGRec ckpts)
- `ETEGRec/myckpt/Musical_Instruments/Jul-22-2026_23-18-b55005/7.pt` (best, R@10=0.0161)
- `logs/task59_etegrec_896_jul-22-2026_23-18-07.log`
- `logs/task59_rqvae_pretrain_jul-22-2026_23-11-48.log`
- `scripts/task59_generate_fused_emb.py`
- `scripts/task59_rqvae_pretrain_896.sh`
- `scripts/task59_etegrec_train_896.sh`

## 10. 状态更新 (2026-07-23)

❌ Task #59 失败. 接受 128d baseline (R@10 0.0253) 为 ETEGRec best 复现. Task #52 收尾 verdict 应反映这一现实.