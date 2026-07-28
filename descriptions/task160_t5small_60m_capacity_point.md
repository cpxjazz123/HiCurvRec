# Task #160 — T5-small 60M (d_model=512, 6+6 layers) — T5 容量 ladder 第 3 点

> **任务目的**: 在 4 档 T5 容量扫描 (5.5M → 12M → 60M → 220M) 中提供 **60M 中点**, 验证 R@10 是否随 T5 容量单调上升.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动 (launcher 写好, 等 GPU 2 跑)

---

## 1. 背景

承接 Task #159 T5-mini 12M (12M 档), Task #156 T5-small 5.5M (5.5M 档), Task #157 T5-base 220M (220M 档). 本任务是 4 档 ladder 的第 3 点 — 标准 T5-small from-scratch (~60M params).

承接用户 2026-07-24 "**帮我同步跑多个, 仅仅t5容量不同的版本, 看看结果如何**" 的请求.

---

## 2. 实验设计

**变量 (唯一)**: T5 容量 = ~60M params (d_model=512, 6+6 layers, d_ff=2048, 8 heads × 64 d_kv, vocab_size=1025)

**保持不变**:
- codebook: Task #156 `_t5_rqvae_code_default.npy` (β=0.25, codebook_size=[32,64,256,1], sk=0.5)
- 数据集: Instruments
- Stage 3 超参: 200 epoch, batch 256, lr 1e-4, dropout 0.1, early stop 20, seed 42, beam_size=20
- Stage 4 评估: topk [5, 10, 20], test parquet leave-one-out

**启动命令** (Stage 3):
```bash
bash scripts/task160_hgrec_t5small60m_stage3.sh
```

**Stage 4**:
```bash
bash scripts/task160_hgrec_t5small60m_stage4_eval.sh
```

---

## 3. 决策触发

| 观察 | 决策 |
|------|------|
| R@10 ∈ [0.105, 0.115] (与 Task #84 / #159 持平) | T5 12M/60M 都不够, 进一步看 220M 是否突破 |
| R@10 ≥ 0.115 (Task #84 +6%) | T5 60M 边际收益明显, ladder 上行 |
| R@10 跑不动 (NaN / OOM) | 缩小 batch 到 128 (L40S 46GB 仍有余) |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Stage 3 训练 (~60M, 200 ep) | ~22-25h on L40S (≈ Task #84 5.5M × 11x = 22h, 线性外推) |
| Stage 4 推断 + R@K 评估 | ~15 min |
| 总计 | ~25h |

---

## 5. 风险与缓解

**风险 1**: 60M 训练时间 ~25h, GPU 2 占用 ~1 天, 期间不能被抢占 (R7 保护).
**风险 2**: 训练突发 OOM — d_model=512 batch=256 在 L40S 46GB 上 ~25GB, 余 21GB, 风险较低; 若 OOM 减 batch 到 128.
**风险 3**: 训练丢失无 ckpt — 已按 R12 修补 fork (`save_limit=1`).

---

## 6. 完成度跟踪

- [x] R9 编号 #160 登记 (descriptions/task160_t5small_60m_capacity_point.md)
- [x] 写 launcher (scripts/task160_hgrec_t5small60m_stage3.sh + stage4_eval.sh)
- [x] bash -n 验证语法
- [ ] 启动 Stage 3 (GPU 2)
- [ ] Stage 4 评估
- [ ] 写 verdict `verdicts/task160_t5small_60m_capacity_point_result.md`

---

## 7. R11.3 决策点

1. **d_model=512 + heads=8 × d_kv=64**: 8×64=512=d_model ✅ exact match, 无重投影.
2. **GPU 2 (空闲)**: R7 合规 (Task #157 GPU 0, Task #156 GPU 3, #159 GPU 1).

---

result: (任务执行中)
