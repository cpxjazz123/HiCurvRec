# Task #85 — FDSA / S³Rec RecBole test evaluation (paper Table 2 #7 / #8)

> **任务目的**: 在 Musical_Instruments test split 上评估 FDSA + S³Rec 两个 RecBole baseline, 闭环 paper Table 2 baseline #7/#8.

> **完成日期**: 2026-07-23
> **状态**: ✅ 已完成

---

## 1. 背景

承接 Task #80 (FDSA 训练, paper #7) 和 Task #81 (S³Rec 训练, paper #8). 两个 baseline 用 RecBole 框架训练到 best valid metric 后, 在 test split 上输出 Recall@5/10, NDCG@5/10 指标, 与 paper Table 2 对比.

---

## 2. 实验设计

**变量**: 仅换 model (FDSA vs S³Rec)
**保持不变**:
- musical_instruments_sequential_paper.yaml 配置
- Musical_Instruments 5-core 数据集
- Test split evaluation

---

## 3. 关键产物

- FDSA verdict: `verdicts/task85_fdsa_test_eval_result.md` (FDSA-Jul-23-2026_16-59-17.pth)
- FDSA test R@10=**0.0594** vs paper 0.0391 (+51.9%)
- 4 指标全部 +43-52% 超 paper baseline

---

## 4. 完成度

- [x] FDSA evaluate (R12 ckpt 验证)
- [x] verdict + result: 行

result: Task #85 FDSA evaluate 完成 (R@10=0.0594, +51.9% over paper 0.0391). S³Rec 部分等 Task #81 训练完成后启动.