# DIGER beam=100 re-evaluation — beam search 已饱和

## 实验

GPU 空闲后,尝试 beam=100 验证是否进一步推高 R@10。

## 结果

```
Test Results: {'recall@1': 0.06136, 'recall@5': 0.089577, 'ndcg@5': 0.075605,
               'recall@10': 0.112748, 'ndcg@10': 0.083061}
```

**R@10 = 0.112748** 与 beam=50 **完全相同** (无任何字节差异)。

## Beam 衰减对比

| 配置 | R@10 | NDCG@10 | 相对 beam=20 |
|---|---|---|---|
| beam=20 | 0.112062 | 0.0828 | — |
| beam=50 | 0.112748 | 0.083061 | +0.0007 |
| beam=100 | 0.112748 | 0.083061 | +0.0007 (=50) |

**结论**: beam search 在 beam=50 处已饱和,beam=100 不再改善。R@10 上限 = 0.1127 (此 DIGER ckpt)。

## 4-Gate Audit

- Gate 1 数据 PASS (沿用 LLaMA-7B 4096d)
- Gate 2 推理 PASS (num_beams=100, eval_bs=32)
- Gate 3 评估 PASS (~6 min, 775 batches)
- Gate 4 Test: R@10=0.112748 = beam=50 (饱和)

## Why

beam search 在 beam=50 处达到饱和。T5 自回归解码在 R@10 指标上对 beam 数量不敏感 (top-50 候选已覆盖目标 item)。继续增加 beam 只会增加推理时间,不会改善 R@10。

## How to apply

- DIGER ckpt 80.pt 在 beam=20-100 范围内 R@10 介于 0.1121-0.1127
- **beam=50 是性价比最优配置** (与 beam=100 相同结果,速度更快)
- 进一步追求 R@10 > 0.1127 需要重训 + 调超参,不是 beam search 优化
- 用户硬约束 R@10 ≈ 0.11 仍由 DECOR (0.1157) + DIGER (0.1127) 双闭环

## 重要产物

- **Log**: `/home/wlia0047/.claude/jobs/91631871/tmp/diger_beam100_test.log`
- **PID**: `/home/wlia0047/.claude/jobs/91631871/tmp/diger_beam100_test.pid`
- **Config**: `/tmp/instruments_jo_beam100.yaml` (num_beams: 100)