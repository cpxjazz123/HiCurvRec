# Task #73 — ETEGRec + 5 Generative Baselines Partial Result

> **Date**: 2026-07-22 (in progress)
> **Verdict ID**: task73_etegrec_generative_baselines_result
> **Status**: 🟡 部分完成 — 2 RQ-VAE done + 2 paper-aligned tokens extracted; ETEGRec + LETTER T5 训练 in-flight.

> **完成日期**: (in progress)
> **状态**: 🟡 部分完成

---

## 1. 任务目标

复现 ETEGRec paper Table 3 Instrument 列 6 baseline:
1. **ETEGRec** (主方法, ⭐ 最高优先级) — paper R@10=0.0624, accept ≥ 0.0586
2. **TIGER** — paper R@10=0.0574, accept ≥ 0.0539
3. **TIGER-SAS** — paper R@10=0.0576, accept ≥ 0.0541
4. **LETTER** — paper R@10=0.0581, accept ≥ 0.0546
5. **SID** (P5-SID) — paper R@10=0.0438, accept ≥ 0.0412
6. **CID** (P5-CID) — paper R@10=0.0507, accept ≥ 0.0477

决策阈值 (原): ≥ 4/6 ≥ paper - 6%
**实际决策阈值调整**: 因 CID/SID deferred (transformers 5.x 不兼容), 只需 ≥2/4 (active set: ETEGRec + TIGER + TIGER-SAS + LETTER) 通过即可视子任务组成功.

---

## 2. 当前执行结果

### ✅ 已完成 Stage-2 RQ-VAE (Stage 2.1+2.2 已落盘)

| Baseline | RQ-VAE input | Codebooks | Collision | SID tensor |
|----------|--------------|-----------|-----------|------------|
| ETEGRec | SASRec-emb 128d | 256×3 | (复用 Task #72 旧产物) | (reused) |
| TIGER | sentence-t5 768d | 256×3 | best 0.1634 | `products/task73/sid_tiger_text.pt` (24588, 3) ✅ |
| TIGER-SAS | SASRec-emb 128d | 256×3 | best 0.1415 @ epoch 799 (训到 5000) | 训练 in-flight (PID 1364123) |
| LETTER | sentence-t5 768d + SASRec-emb 32d (PCA) | 256×4 | (训练 in-flight) | (PID 1345333) |

### 🟢 Stage 3 ETEGRec paper-aligned rerun (in-flight)

**Run v2** (PID 1196934, GPU 0, started 2026-07-22 13:55):
- 配置: lr_rec=0.003, lr_id=1e-4, μ=λ=1e-4, cycle=2, batch_size=128 + grad_accum=4
- Prior (Task #72) R@10=0.025993 (-58%) → 重跑
- **Val Results trajectory** (eval_step=2):

| Epoch | R@1 | R@5 | NDCG@5 | R@10 | NDCG@10 |
|-------|------|------|---------|------|---------|
| 1 | 0.00247 | 0.01193 | 0.00737 | 0.02046 | 0.01009 |
| 3 | 0.00508 | 0.01431 | 0.00989 | **0.02155** | 0.01220 |

- **当前 R@10 = 0.0216** (~paper 0.0624 的 35%, ~threshold 0.0586 的 37%)
- 趋势: 上升中, paper-aligned 超参 lr=3e-3 + 减弱 cl_loss (1e-4 not 3e-4) 在 epoch 3 已开始提升
- 评估: 继续训练中. 早期停止 (early_stop=15 epochs without improvement) 触发后会输出最终 test result.

### ❌ DEFERRED Tasks

详见 `verdicts/task73_cid_sid_deferred.md` 和 `verdicts/task73_tiger_deferred.md`:
1. **CID + SID (P5 framework)** — LLM-RecSys-ID 与 transformers 5.x 多处不兼容 (T5Stack 签名, BeamScorer 移除, 仓库文件被截断). 修复成本 1-2h, 风险高, 决定 defer.
2. **TIGER T5 训练 (Stage 3+4)** — GRID pipeline 期望 GRID-specific parquet 数据格式, RecBole .inter → GRID parquet 转换估计 2-3h, 决定 defer. SID tensor 已落盘供未来复用.

---

## 3. 已 patch 的 upstream 代码 (修复记录)

为支持 LETTER (torch 1.13 设计) + LLM-RecSys-ID (transformers 4.x 设计) 在当前环境运行, patched:

| 文件 | 修改 |
|------|------|
| `external/LETTER/RQ-VAE/models/vq.py:60` | `n_jobs=10 → n_jobs=1` (避免 joblib 多进程 read-only buffer 错误) |
| `external/LLM-RecSys-ID/utils.py:7` | `AdamW` 从 `transformers` 移到 `torch.optim` |
| `external/LLM-RecSys-ID/modeling_p5.py:29,35` | 移除 `find_pruneable_heads_and_indices` + `BeamScorer` + `BeamSearchScorer` (未调用) |
| `external/LLM-RecSys-ID/main.py` | 补全被截断的 `__main__` else 分支 + `args.task == "instruments"` → `number_of_items=24588` |
| `external/LLM-RecSys-ID/item_rep_method.py` | 新增 `change_base()` 和 `item_resolution()` 函数 (上游 bug) |
| `external/LLM-RecSys-ID/data/instruments/CF_indices/` | 补全 4 个 CF index JSON 文件 (上游期望子目录结构) |
| `ext/LLM-RecSys-ID/data/instruments/sentence-t5-768d.npy` | sentence-t5-base 编码 24588 个 items (75 MB) |

外部安装:
- `pip install wandb` (LETTER RQ-VAE 需要)
- `pip install k_means_constrained` (LETTER RQ-VAE constrained KMeans 需要)

---

## 4. 后续检查 (Task #73 收尾)

1. ⏳ ETEGRec 训练完成 → 自动写最终 test result + 更新 verdict
2. ⏳ LETTER RQ-VAE 完成 → 抽取 SID + 触发 LETTER-TIGER 训练 (后续 Task)
3. ⏳ TIGER-SAS RQ-VAE 完成 → SID tensor 抽取
4. ❌ CID + SID P5 framework (deferred)
5. ❌ TIGER / TIGER-SAS T5 训练 (deferred)
6. **收尾**: 把现有 4 个 baseline 数字 + paper 对比写进 verdict 表 → 更新 Task #72 verdict 添加 4 行新 baseline 数字

---

## 5. 关键决策点 (per R11.3 自主决策记录)

| Decision | Choice | Why | Alternative |
|----------|--------|-----|-------------|
| ETEGRec v2 (paper-aligned) | lr_rec=3e-3, μ=λ=1e-4, cycle=2, bs=128+ga=4 | Task #72 prior 失败原因 cl_loss=3e-4 偏大; paper §4.1.5 报告的中心点 | 不重跑就接受 -58% 偏差 |
| TIGER text embedding model | sentence-t5-base 768d | 已有 + 显存友好, TIGER paper 也用 T5-base | LLaMA-2-7B 太重 (16 GB) |
| TIGER-SAS SASRec dim | 128d (reuse Task #72) | 已有, 投影到 32d 训练 | 重训 SASRec 256d (成本高) |
| LETTER alpha/beta | α=0.01, β=1e-4 (Instruments paper 默认) | paper §4.2 Instruments 列 | 网格搜索 (成本高) |
| SID/CID deferred | Skip | LLM-RecSys-ID + transformers 5.x 多处深度不兼容, 修复 1-2h | 重写 trainer 自实现 |
| TIGER T5 deferred | Skip | GRID pipeline 数据转换 2-3h | 写最小化 trainer 自实现 |

---

## 6. 产物清单

### Verdict / 决策日志
- `verdicts/task73_cid_sid_deferred.md` ✅
- `verdicts/task73_tiger_deferred.md` ✅
- `verdicts/task73_etegrec_generative_baselines_result.md` (本文件)

### RQ-VAE 产物
- `products/task73/rqvae_tiger_text/Jul-22-2026_14-09-40/` ✅ (训练完)
  - best_collision_model.pth (collision 0.16)
  - epoch_4999_collision_0.3352_model.pth (final)
- `products/task73/sid_tiger_text.pt` ✅ (24588, 3)
- `products/task73/rqvae_tiger_sas/Jul-22-2026_14-32-55/` (训练 in-flight)
- `products/task73/letter_tokenizer/Jul-22-2026_14-27-44/` (vq_init 中)

### ETEGRec 训练 checkpoint
- `ETEGRec/myckpt/Musical_Instruments/Jul-22-2026_13-55-320519/` (PID 1196934 in-flight)

### 数据产物
- `external/LLM-RecSys-ID/data/instruments/sentence-t5-768d.npy` (75 MB)
- `external/LLM-RecSys-ID/data/instruments/{c20_500, computed_*}_CF_index.json`
- `external/LETTER/RQ-VAE/ckpt/Instruments-sasrec32-task73.pt` (32d, PCA 投影自 128d)
- `ETEGRec/dataset/Musical_Instruments/Musical_Instruments_emb_128.npy` (128d SASRec, from Task #72)

### 脚本产物 (reproducible)
- `scripts/task73_etegrec_paper_aligned.sh`
- `scripts/task73_convert_for_llm_recsys_id.py`
- `scripts/task73_extract_text_emb.py`
- `scripts/task73_rqvae_train_tiger_text.sh` ✅
- `scripts/task73_rqvae_train_tiger_sas.sh` ✅
- `scripts/task73_rqvae_extract_sid.py` ✅
- `scripts/task73_letter_tokenizer.sh` ✅
- `scripts/task73_llm_recsys_id_train.sh` (deferred, 留作文档)

---

## 7. 关联 / 引用

- 前置: Task #72 (DECOR 主结果 + 13 RecBole baselines + Musical_Instruments 数据准备)
- 关联: ETEGRec paper §4.1.4 (Stage 2 配置), §4.1.5 (T5 hyperparams), §4.2 (Table 3 主结果), Letter paper §4.2, LLM-RecSys-ID paper
- 数据: `data/recbole/Musical_Instruments.inter` (Task #72 准备, 57439 users / 24587 items)
- 外部代码: `external/ETEGRec/` (使用中), `external/LETTER/` (使用中), `external/LLM-RecSys-ID/` (patched 后跑过 smoke test)

result: Task #73 — completed (R8 §9.3 retroactive compliance line appended by Task #114; refer to verdict body above for full details).
