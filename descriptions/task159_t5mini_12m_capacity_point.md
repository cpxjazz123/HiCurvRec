# Task #159 — T5-mini 12M (d_model=256, 4+4 layers) — T5 容量 ladder 第 2 点

> **任务目的**: 在 4 档 T5 容量扫描 (5.5M → 12M → 60M → 220M) 中提供 **12M 中间点**, 验证 R@10 是否随 T5 容量单调上升. 若单调 → T5 容量是真正瓶颈; 若平台 → paper gap #120 唯一真因 (paper 内部代码不可见) 进一步确认.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动 (launcher 写好, 等 GPU 1 跑)

---

## 1. 背景

承接 **Task #120 paper gap 唯一真因 = paper 内部代码 ≠ paper github 代码** (paper 报告 0.1315 来自不可见代码).

承接 **Task #87 paper-aligned 8 baseline 综合 ranking**: 8/8 baseline 都低于 paper 18-61%, 相对 ranking 保留但绝对数字不保留.

承接用户 2026-07-24 "**帮我同步跑多个, 仅仅t5容量不同的版本, 看看结果如何**" 的请求.

用户判断: 现在无法排除 T5 容量仍是瓶颈, 必须做 controlled ablation (仅 T5 容量变化, 其他变量保持不变) 量化 T5 容量对 R@10 的边际影响.

---

## 2. 实验设计

**变量 (唯一)**: T5 容量 = ~12M params (d_model=256, 4+4 layers, d_ff=1024, 6 heads × 64 d_kv, vocab_size=1025)

**保持不变**:
- codebook: Task #156 `_t5_rqvae_code_default.npy` (β=0.25, codebook_size=[32,64,256,1], sk=0.5)
- 数据集: Instruments (Amazon Musical Instruments, 24772 users / 9922 items / 206153 interactions, 5-core 100%)
- Stage 1 sentence-t5-base 768-dim item embedding
- Stage 3 超参: 200 epoch, batch 256, lr 1e-4, dropout 0.1, early stop 20, seed 42, beam_size=20
- Stage 4 评估: topk [5, 10, 20], test parquet leave-one-out

**启动命令** (Stage 3):
```bash
bash scripts/task159_hgrec_t5mini12m_stage3.sh
```

**Stage 4**:
```bash
bash scripts/task159_hgrec_t5mini12m_stage4_eval.sh
```

---

## 3. 决策触发

| 观察 | 决策 |
|------|------|
| R@10 ∈ [0.105, 0.115] (与 Task #84 持平) | T5 12M 不够; 转入 Task #160 (60M) 看是否上升 |
| R@10 ≥ 0.115 (超过 Task #84 +6%) | T5 12M 边际收益存在, 跑完看 60M 是否进一步 |
| R@10 跑不动 (NaN / collapse) | T5 config 改为 (4+4 layers, d_model=192) |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Stage 3 训练 (~12M, 200 ep) | ~4-5h on L40S |
| Stage 4 推断 + R@K 评估 | ~10 min |
| 总计 | ~5h |

---

## 5. 风险与缓解

**风险 1**: d_model=256, 6 heads × 64 d_kv = 384 d_kv > d_model=256, 必须设置 d_kv=64 而非 64 (实际应是 heads × d_kv ≤ d_model) → 已确认 fork script d_kv=64, 6×64=384, 但 d_model=256 < 384 → **触发 T5 assert d_kv × heads == d_model**. 缓解: 用 d_kv=64, num_heads=4 (4×64=256=d_model) — 实际上 fork 已设为 6 heads × 64 d_kv 没关系, T5 会自动 assert. 看 forks 错误处理.

**风险 2**: T5 12M 训练丢失 (无 ckpt) — 已按 R12 修补 fork (`save_limit=1`).

---

## 6. 完成度跟踪

- [x] R9 编号 #159 登记 (descriptions/task159_t5mini_12m_capacity_point.md)
- [x] 写 launcher (scripts/task159_hgrec_t5mini12m_stage3.sh + stage4_eval.sh)
- [x] bash -n 验证语法
- [ ] 启动 Stage 3 (GPU 1)
- [ ] Stage 4 评估
- [ ] 写 verdict `verdicts/task159_t5mini_12m_capacity_point_result.md`
- [ ] 综合 ladder (合并 4 档 v.s. paper 0.1315)

---

## 7. R11.3 决策点

1. **d_model=256 + heads=6**: 6×64=384 ≠ 256. T5 HuggingFace 允许 heads × d_kv ≠ d_model (会重投影), 我们这次遵循 fork (heads=6, d_kv=64). 若 transform 失败, 退化到 heads=4, d_kv=64, 4×64=256=d_model.
2. **复用 Task #156 codebook**: 一致 codebook → 4 档 ladder 唯一变量 = T5 容量.
3. **GPU 1 (空闲)**: R7 合规 (Task #157 跑 GPU 0 = 99%, Task #156 跑 GPU 3 = 89%, 1/2 0%/0%).

---

result: (任务执行中)
