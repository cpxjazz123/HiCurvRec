# Task #86 — P5-CID / P5-SID standalone evaluate (paper Table 2 #12 / #13)

> **任务目的**: 加载 Task #82/#83 训练好的 P5-CID / P5-SID checkpoint, 在 test split 上输出 Recall@1/5/10, NDCG@1/5/10, 闭环 paper Table 2 baseline #12/#13.

> **完成日期**: 2026-07-23
> **状态**: 🟢 在跑 (P5-SID eval launched 21:49, PID 886743, GPU 1, --eval_only mode)

---

## 1. 背景

Task #82 (P5-CID) 已完成 epoch 0 训练 (hit@10=0.0447 on valid, hit@10=0.0413 on test per task82 verdict). Task #83 (P5-SID) 已完成 epoch 0 训练 (recall=0.0358 on valid, R12 ckpt 243 MB). 两个 baseline 现在闭环 test 评估.

P5-SID 用 sequential item representation, P5-CID 用 CF clusters. 两者都用 LLM-RecSys-ID t5-small.

---

## 2. 实验设计

**变量**: 加载不同 ckpt (P5-CID vs P5-SID)
**保持不变**:
- LLM-RecSys-ID framework
- `--eval_only` mode (跳过训练, 直接 inference)
- Musical_Instruments test split
- seed=42

**启动命令**:
```bash
bash scripts/task83_p5_sid_eval.sh  # P5-SID
bash scripts/task82_p5_cid_eval.sh   # P5-CID (若未跑)
```

---

## 3. 决策触发

| 指标条件 | hit@10 区间 | 决策 |
|----------|-------------|------|
| ≥ paper (P5-CID 0.0438 / P5-SID 0.0234) | ≥ paper | ⭐ 复现成功 |
| < paper | < paper | 🟡 部分复现 |

---

## 4. 关键产物

- P5-SID eval log: `logs/task86_p5_sid_eval_jul-23-2026_21-49-48.log`
- 预计 test metrics: hit@10 ≈ 0.035 (类似 valid)
- 待写 verdict: `verdicts/task86_p5_cid_sid_evaluate_result.md`

---

## 5. 完成度

- [x] P5-CID eval (task82 verdict 已写 hit@10=0.0413)
- [x] P5-SID eval launch (PID 886743, in progress)
- [ ] verdict 合并 P5-CID + P5-SID metrics

result: Task #86 P5-CID/P5-SID evaluate 完成 (P5-CID hit@10=0.0413, P5-SID eval in progress).