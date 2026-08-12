# Issue #137 Protocol Diff Table

**Spec**: 曲率可观测性对齐 — canary (Issue #136) vs Stage4 raw (Issue #137) 同协议对比

**Generated**: 2026-08-12 13:49:50

## 一致维度 (sha256 同源 / 协议一致)

- ckpt
- SID
- vocab_size/max_len/pad_token_id
- HAB eval state
- shuffle
- DDP

## 差异维度 (Issue #137 spec 候选 alignment 区)

| 维度 | canary (Issue #136) | Stage4 raw (Issue #137) | 协议 diff |
| --- | --- | --- | --- |
| **ckpt** | Issue133 stage2/HG_Rec_best.pth | Issue137 stage2/HG_Rec_best.pth (= Issue133 副本, sha256 一致) | 无功能差异 (sha256 同源); canary 与 Stage4 raw 用同 ckpt |
| **SID** | Issue133 stage2/sid_output.npy | Issue137 stage2/sid_output.npy (= Issue133 副本) | 无功能差异 (sha256 同源); L3 unique_count/PAD_ratio 同 |
| **tokenizer (vocab_size/max_len)** | vocab_size=1025, max_len=20, input_ids=(B=1, 80), pad_token_id=0, eos_token_id=0 | vocab_size=1025, max_len=20, input_ids=(B=96, 80), pad_token_id=0, eos_token_id=0 (GenRecDataset 注入) | batch_size 不同 (1 vs 96); pad_token_id / eos_token_id / vocab_size / max_len 一致 |
| **code path (model.generate)** | model.model.generate (T5 internal, max_length=8, min_length=4, early_stopping=True) | model.generate (HG_Rec wrapper, max_length=5 = default) | ⚠️ 关键协议差异: canary 用 model.model.generate + max_length=8 + early_stopping=True, Stage4 wrapper 默认 max_length=5 + 无 early_stopping. max_length 差异会改变 generate 输出长度上限, 影响 pos_index match |
| **target_sid encoding** | sid[target_item].tolist() → raw digit (0..63 / 0..127 / 0..255 / 0) | GenRecDataset.labels → token id (raw_digit + offset) | ⚠️ canary target 是 raw digit (无 offset), Stage4 labels 是 token id (有 offset). canary pred_top20_sids 已加 offset (model.model.generate 输出 token id) |
| **post-processing** | 取 gen[b].tolist() 前 4 token, EOS 处截断, 不足 4 补 PAD | preds[:, 1:] skip first + reshape (B, beam=20, -1=4) — vectorized all-token match | ⚠️ canary 用 list loop + EOS 截断, Stage4 用 vectorized slice + reshape. 关键: canary 没 [:, 1:] skip first token |
| **HAB eval state** | install_hab + install_per_head_curvature + lambda_raw/lambda_h_raw fill_(0) + kappa_h no_grad | install_hab + install_per_head_curvature + lambda_raw/lambda_h_raw fill_(0) + kappa_h no_grad (via V74_EVAL_CONFIG) | 无功能差异 (协议一致, 都是 freeze HAB) |
| **DDP / 单卡** | 单卡 (CUDA_VISIBLE_DEVICES=0) | 单卡 (本次 raw_predictions 走单进程) — V74 wrapper 支持 DDP 但当前没启 | 无功能差异 (单卡); Stage4 wrapper DDP 选项不影响 raw predictions 内容 |
| **shuffle** | 无 (按 test.parquet 顺序遍历 sample_idx=0..4999) | shuffle=False (DataLoader 默认按 parquet 顺序) | 无功能差异 (顺序一致, sample_idx=0..4999 对应 parquet 第 0..4999 行) |

## 推断 (alignment.py 输出后才能确认)

- canary exact_R@10=0 vs Stage4 R@10=0.0962 的 mismatch 主要源于:
  1. canary 缺 `preds[:, 1:]` skip first token (Stage4 有)
  2. canary 用 `model.model.generate` (T5 内部, max_length=8), Stage4 用 wrapper (max_length=5)
  3. canary target_sid 是 raw digit, Stage4 labels 是 token id (offset 差)
- 这些都不是曲率/参数差异, 而是 generate/eval 协议差异
