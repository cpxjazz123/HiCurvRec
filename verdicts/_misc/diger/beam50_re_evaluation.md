# DIGER beam=50 re-evaluation 结果

## 背景

GPU 0 之前持续被遗留 LETTER RQ-VAE NO-GO 训练进程阻塞 4+ 小时,本次验证:
1. GPU 空闲后立即可重新评估现有最佳 ckpt
2. beam=20 → beam=50 是否能推高 R@10

## 实验配置

- **Ckpt**: `/home/wlia0047/ar57/wenyu/DIGER/myckpt/instruments/Aug-05-2026_12-58-dbd741/80.pt` (181 MB, epoch 80 best)
- **Config**: `instruments_jo.yaml` (FrqUD joint optimization 配置)
- **改动**: `num_beams: 20 → 50`
- **GPU**: 0 (空闲, 0% util / 0 MB / 0 进程)
- **环境修复**: `attn_implementation='sdpa'` → `'eager'` (transformers 版本兼容)

## 4-Gate Audit

### Gate 1 — 数据 + Embedding: PASS (沿用)
- Instruments.emb-llama.npy (LLaMA-7B 4096d) ✅
- RQ-VAE ckpt 沿用 0.1121 run 同款 ✅

### Gate 2 — 推理配置: PASS
- num_beams 20 → 50 (Decode 候选数 ×2.5)
- eval_batch_size 32 保持
- 实际耗时: ~3 min (775 batches, 5.5 it/s avg)

### Gate 3 — 评估: PASS
- test_only 模式加载 best ckpt + eager attention
- 24772 test users, beam search 生成 10 个 item

### Gate 4 — Stage Test: R@10=0.1127 (+0.0007 vs beam=20)

```
Test Results: {
  'recall@1': 0.06136,        # 同
  'recall@5': 0.089577,       # +0.000121 vs 0.089456
  'ndcg@5': 0.075605,         # +0.000077 vs 0.075528
  'recall@10': 0.112748,      # +0.000686 vs 0.112062  ← 主指标
  'ndcg@10': 0.083061         # +0.000261 vs 0.0828
}
```

## 结果分析

**beam=50 → R@10=0.1127** 相比 beam=20 → R@10=0.1121,**提升 +0.0007 (+0.6%)**。提升边际,符合 beam search 的典型衰减规律 (beam 20→50 通常 +0.5-1%)。

## 与基线对比

| 配置 | Test R@10 | vs HG-Rec | vs Target 0.11 |
|---|---|---|---|
| DIGER beam=50 | **0.1127** | +10.1% | +2.5% |
| DIGER beam=20 (原) | 0.1121 | +9.5% | +1.9% |
| DECOR | 0.1157 | +13.0% | +5.2% |
| HG-Rec baseline | 0.1024 | — | -7% |
| ETEGRec | 0.0763 | -25.5% | -30.6% |
| LETTER-TIGER | 0.0581 | -43.3% | -47.2% |

## Why

GPU 阻塞解除后,立即尝试快速改进现有 DIGER ckpt (无需重新训练)。beam=20 → beam=50 推高 R@10 从 0.1121 → 0.1127 (+0.0007),边际改善符合 beam search 衰减规律。约束 R@10 ≈ 0.11 仍由 DIGER (0.1127) + DECOR (0.1157) 双达成。

## How to apply

- GPU 空闲后**可立即执行快速 re-evaluation** (~3 min),无需重新训练
- beam=50 边际改善 +0.0007 vs beam=20,不值得作为标准配置
- 进一步追求 R@10 > 0.115 需要: 调 DIGER 超参 (multi-seed / lr sweep / wd) 或重训更长 epoch
- DIGER + DECOR 双方法已闭环,任务完成

## 重要文件

- **Log**: `/home/wlia0047/.claude/jobs/91631871/tmp/diger_beam50_test.log`
- **PID**: `/home/wlia0047/.claude/jobs/91631871/tmp/diger_beam50_test.pid`
- **Config 改动**: `/tmp/instruments_jo_beam50.yaml` (num_beams: 50)
- **Code 改动**: `/home/wlia0047/ar57/wenyu/DIGER/_test_eval.py` (attn_implementation='eager')