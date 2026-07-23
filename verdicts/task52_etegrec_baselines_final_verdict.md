# Task #52 (FINAL) — ETEGRec + 5 Generative Baselines 复现终判 (Musical_Instruments)

> **完成日期**: 2026-07-23
> **任务**: Task #73 主线 — ETEGRec + TIGER + TIGER-SAS + LETTER + SID + CID — 6 baseline 完整复现 (含整合 verdict + 更新 Task #72)
> **状态**: 🟡 **部分完成 — 1 个完整 baseline 成功 (LETTER), ETEGRec 部分成功, TIGER/P5 未完成**

---

## 1. 一句话结论

**6 个 generative baseline 在 Musical_Instruments 上: 1 个完整成功 (LETTER R@10=0.0997), 1 个部分成功 (ETEGRec 128d R@10=0.0253), 4 个未完成 (TIGER, TIGER-SAS, P5-SID, P5-CID). 用户的"反思 paper 差距"原则触发 896d 实验失败后回退到诚实标注现状.**

---

## 2. 6 baselines 状态表

| # | Baseline | Task | 训练 | 推断 | R@10 | 状态 |
|---|---------|------|------|------|------|------|
| 1 | **ETEGRec 128d** | #48+#73 paper_exact | ✅ 完成 | ✅ 复现 0.0253 | **0.0253** | 🟡 partial (-59% vs paper 0.0624) |
| 2 | **ETEGRec 896d fused** | #59 | ✅ 完成 (1:32) | ✅ 复现 0.0161 | 0.0161 | ❌ failed (比 128d 更差) |
| 3 | TIGER | #49 | ❌ 仅 RQ-VAE | ❌ 未跑 | - | ❌ incomplete |
| 4 | TIGER-SAS | #49 | ❌ 仅 RQ-VAE | ❌ 未跑 | - | ❌ incomplete |
| 5 | LETTER-TIGER | #50+#61 | ✅ 完成 (79 epoch) | ✅ 0.0997 | **0.0997** | ✅ **done (+72% vs paper 0.0581)** |
| 6 | P5-SID | #51+#60 | ❌ P5 框架崩 | ❌ 未跑 | - | ❌ incomplete |
| 7 | P5-CID | #51+#60 | ❌ P5 框架崩 | ❌ 推断崩 | - | ❌ incomplete |

> ⚠️ **诚实标注**: 原始 Task #73 期望 6 个 baseline, 实际完整复现的只有 1 个 (LETTER), 1 个部分 (ETEGRec 128d). TIGER / TIGER-SAS / P5-SID / P5-CID **未真正完成**. 这是 Task #47-#60 5 个子任务的 honest 总结.

---

## 3. 详细结果

### 3.1 ETEGRec 128d (Task #48 + Task #73 paper_exact) — 🟡 partial

- **训练**: cycle=2 + warmup_steps=8000 + warm_epoch=10 + early_stop=15, 400 epoch budget
- **数据集**: Musical_Instruments (57439 users / 24587 items / 511837 interactions)
- **Embedding**: 128d SASRec-only (paper 默认)
- **Best R@10**: 0.0253 (epoch 15)
- **paper 目标**: 0.0624
- **偏差**: -59%
- **结论**: 比 paper 低 59%. RQ-VAE collision 81% (item 共享 SID). cycle=2 训练在我们的 budget 内未收敛到 paper 报告值.

### 3.2 ETEGRec 896d (Task #59) — ❌ failed

- **目标**: 修复 128d 的 -59% 偏差. 假设: 128d SASRec-only 缺 text 信息, 改用 896d fused (sentence-t5 768d + SASRec 128d).
- **结果**: R@10 0.0113 (avg) / 0.0161 (best epoch 7), 比 128d 0.0253 **反而低 -54%**.
- **RQ-VAE**: ✅ 编码质量大幅改善 (collision 81% → 9.33%, code balance L3 0.69 → 0.85)
- **训练**: ❌ KL loss 0.002 vs 128d 78.7 — 10000× 差异, ID cycle 没学到结构化 latent
- **否证假设**: "提升 RQ-VAE 编码质量 → 提升 ETEGRec R@10" 是**假**.
- **终止**: 1:32 elapsed, epoch 11 显示下降趋势. 按 R11 + "反思 paper 差距"原则停止.
- **详细**: 见 `verdicts/task59_etegrec_896d_verdict.md`

### 3.3 TIGER + TIGER-SAS (Task #49) — ❌ incomplete

- **现状**: Task #49 标记 "completed" 但实际**只完成 Stage 2 (RQ-VAE training + SID inference)**, 产出:
  - `products/task73/sid_tiger_text.pt` (24588, 3) int64
  - `products/task73/sid_tiger_sas.pt` (24588, 3) int64
  - `products/task73/rqvae_tiger_text/Jul-22-2026_14-09-40/best_collision_model.pth`
  - `products/task73/rqvae_tiger_sas/Jul-22-2026_14-32-55/best_collision_model.pth`
- **缺失**: Stage 3 (TIGER encoder-decoder 训练) + Stage 4 (inference + R@10 评估). 训练产物 0 个.
- **SID tensor 维度**: (N, 3), 但 GRID 框架需要 (N, 4) — 追加 1 列去重 digit 可对齐, 但需要 retrain TIGER encoder-decoder (~6-12h).

### 3.4 LETTER-TIGER (Task #50 + Task #61) — ✅ success

- **训练 (Task #50)**: t5-small + Trie constrained beam search + RQ-VAE 692 tokens, 79 epoch, best eval_loss=1.588 (epoch 59)
- **推断 (Task #61)**: 776 batches × num_beams=20, ~62 min on L40S
- **结果**:
  - **R@1 = 0.0595**
  - **R@5 = 0.0815**
  - **R@10 = 0.0997** ✨
  - **NDCG@5 = 0.0704**
  - **NDCG@10 = 0.0763**
- **vs paper**: Letter paper Instruments R@10 ≈ 0.0581 (ECAI 2024 Table 4). 我们的复现 **+72%** ✅
- **关键 fix**: 推断时 `index_file` 必须与训练一致 (`Instruments.index.epoch5000.alpha0.01-beta0.0001.json`, 692 tokens). 默认 `.index.json` (918 tokens) 会触发 CUDA assert (vocab mismatch).
- **详细**: 见 `verdicts/task61_letter_tiger_inference.md`

### 3.5 P5-SID + P5-CID (Task #51 + Task #60) — ❌ failed (framework broken)

- **P5-SID**: 训练跑过 20 epoch, ckpt 未保存 (P5 framework 不写 best_model_at_end)
- **P5-CID**: 训练 ckpt 找不到 → 推断失败 (`T5Stack.__init__() takes 2 positional arguments but 3 were given`)
- **核心问题**: `external/LLM-RecSys-ID/` 框架使用 `transformers` 4.x T5Stack 接口, 当前 genrec_env 装的是 transformers 5.x — **不兼容**. 修复需要 downgrade transformers 或 fork P5 框架, 工程量太大.
- **诚实结论**: Task #51 (P5 SID + CID) 不能完成. Task #60 (CID inference) 失败已记录.

---

## 4. 与 paper Table 3 对比 (Musical_Instruments R@10)

| Method | Paper R@10 | 我们的 R@10 | 偏差 | 备注 |
|--------|-----------|-----------|------|------|
| ETEGRec | 0.0624 | 0.0253 (128d) / 0.0161 (896d) | -59% / -74% | partial / failed |
| TIGER | 0.0574 | - | - | incomplete |
| TIGER-SAS | 0.0576 | - | - | incomplete |
| LETTER | 0.0581 | **0.0997** | **+72%** ✅ | **唯一完整 + 超 paper** |
| P5-SID | ~0.06 (estimate) | - | - | framework broken |
| P5-CID | ~0.06 (estimate) | - | - | framework broken |

**整体复现率: 1/6 完整成功, 1/6 部分成功, 4/6 未完成.**

---

## 5. 关键决策点 (R11 自主决策汇总)

| 决策 | 选了什么 | 为什么 | 备选 |
|------|---------|-------|------|
| ETEGRec 896d 训练停止 (epoch 11) | 停止 | R@10 0.011 avg < 128d 0.025, 12 epoch 已显停滞趋势 | 继续训练到 100 epoch |
| 接受 128d baseline 为 ETEGRec best | 0.0253 | 128d 在所有尝试中 R@10 最高, 是诚实 best | 重新训练 2048d |
| LETTER fix index_file mismatch | 用 epoch5000 indices (692 tokens) | 训练 ckpt 的 tokenizer 是 692 vocab, 测试必须对齐 | 修改 training data 重新训练 |
| TIGER Stage 3+4 不再启动 | 不启动 | (24588, 3) → (24588, 4) SID 对齐后还需 6-12h 训练, GPU 时间不够 | 重新跑完整 pipeline |
| P5 framework 不再修复 | 不修复 | transformers 5.x 不兼容, 工程量 > 5h, ROI 低 | downgrade transformers |

---

## 6. 产物清单 (Task #73 主线)

### Verdicts
- `verdicts/task59_etegrec_896d_verdict.md` (FAILED)
- `verdicts/task61_letter_tiger_inference.md` (✅ SUCCESS)

### Trained checkpoints
- ETEGRec 128d: `ETEGRec/myckpt/Musical_Instruments/Jul-22-2026_17-02-efdc80/`
- ETEGRec 896d: `ETEGRec/myckpt/Musical_Instruments/Jul-22-2026_23-18-b55005/` (best = epoch 7)
- LETTER-TIGER: `external/LETTER/LETTER-TIGER/ckpt/Instruments/checkpoint-15222/`
- RQ-VAE 896d: `ETEGRec/dataset/Musical_Instruments/256-256-256-128_896din.rqvae.pth`
- TIGER RQ-VAE text/sas: `products/task73/rqvae_tiger_{text,sas}/.../best_collision_model.pth`

### Inference results
- `results/task61_letter_inference/letter_tiger_instruments_test.json` — LETTER R@10=0.0997

### Scripts (12 个)
- `scripts/task59_generate_fused_emb.py`
- `scripts/task59_rqvae_pretrain_896.sh`
- `scripts/task59_etegrec_train_896.sh`
- `scripts/task61_letter_inference.sh`
- (其他 task50/51/60 scripts 略)

---

## 7. 对 Task #72 (DECOR 复现主任务) 的更新

Task #72 主结果 (DECOR + 12 baselines) 已有自己的 verdict. Task #73 generative baselines 是 Task #72 的补充扩展, 结论**不修改** Task #72 的核心结论, 但 Task #52 (本次) verdict 应作为 Task #72 的附录, 标明 generative baseline 部分的状态.

更新位置: `descriptions/task72_*.md` 末尾追加 1 节 "Generative Baseline 扩展状态 (Task #73 / Task #52 final)".

---

## 8. 后续可能任务 (用户决策)

| 候选 | GPU 时间 | ROI | 推荐 |
|------|---------|-----|------|
| TIGER Stage 3+4 完整训练 | 6-12h | 中 (paper 0.0574, 我们 LETTER 0.0997 已超) | ❌ 不推荐 |
| P5 framework 修复 (transformers downgrade) | 5h | 低 (LETTER 已 R@10=0.10, P5 与 LETTER 架构类似) | ❌ 不推荐 |
| ETEGRec 128d 200 epoch 完整训练 | 6h | 中 (可能从 0.025 → 0.03-0.04, 仍低于 paper) | ⚠️ 可选 |
| LETTER 加 T5-base (vs t5-small) | 4h | 高 (可能突破 R@10=0.12+) | ✅ **推荐** |
| TASK #52 自身 → 收尾 done | 5 min | 完成 verdict | ✅ 已做 |

---

## 9. 状态

✅ Task #52 收尾 verdict 已写. Task #73 主线完结 (部分). 等用户下一个任务指示.

---

当前任务已完成，请做下一个任务的指示。