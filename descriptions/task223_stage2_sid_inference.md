# Task #223 — Stage 2 SID 推断 (PCK healthy ckpt)

> **任务目的**: 用 Task #222 早停的 healthy ckpt 推断 SID, 看 Sinkhorn 后的 SID unique 度.

> **完成日期**: 2026-07-26
> **状态**: ✅ 完成

---

## 1. 背景

承接 Task #220 PARTIAL SUCCESS — Per-Codeword κ 在 epoch 29 达到 healthy local min (collision 0.38). 用户决定早停重训 (#222), 用 healthy ckpt 跑 Stage 2 SID 推断, 验证 SID 可用性.

---

## 2. 实验设计

**输入 ckpt**: Task #222 早停 (ep 30) best_collision_model.pth
**输入数据**: HG-Rec/dataset/Instruments/item_emb.parquet (9922 × 768)
**输出 SID shape**: (9922, 4) int array
**Sinkhorn 解码**: 最多 30 轮 + 4th-digit dedup

---

## 3. 决策触发

| 指标 | 通过标准 | 实际 | 决策 |
|------|---------|------|------|
| SID unique rate | ≥ 95% | TBD | 进/不进 Stage 3 |

---

(本文档由 R9-Enforce 层 2 placeholder 填补脚本自动生成, 真实 verdict: verdicts/task223_stage2_sid_inference_result.md)
