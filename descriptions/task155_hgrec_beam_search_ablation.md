# Task #155 — HG-Rec num_beams=50 re-eval (filter_items=False 替代方案)

> **任务目的**: 验证 beam search 大小是否解释 R@10 -22.4% 差距 (作为 filter_items=False 替代)
> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

Task #84 HG-Rec R@10=0.1020, paper=0.1315, Δ -22.4%.
用户提议验证 `filter_items=False` 的影响.
**但 HG-Rec 框架没有 filter_items 概念** (HG-Rec 用 beam search 生成 top-k 直接检查 ground truth, 不像 LETTER 用候选集+历史过滤).

**最接近的替代**: 增大 `num_beams` (从 20 → 50) → 搜索空间更大 → top-k 更准确 → 跟 paper 期望数字更接近 (如果 paper 用更大 beam).

---

## 2. 实验设计

**变量**: `num_beams=20` → `num_beams=50` (Stage 4 inference only)
**保持不变** (跟 Task #84 baseline 完全一致):
- Stage 1/2/3 ckpt 不变 (用 Task #84 best ckpt: `products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/HG_Rec_best.pth`)
- 只改 Stage 4 eval 的 `beam_size` 参数

**启动命令**: `scripts/task155_hgrec_beams50_eval.sh`

**R7 GPU**: GPU 3 (空闲)

---

## 3. 决策触发 (vs Task #84 num_beams=20)

| Task #155 R@10 | Δ vs Task #84 | 解读 |
|----------------|---------------|------|
| [0.1107, 0.1315] | +8% ~ +29% | beam size 是主要根因 ✅ |
| [0.1050, 0.1107) | +3% ~ +8% | beam 是部分根因 |
| [0.0986, 0.1050) | -3% ~ +3% | beam size 影响小, 训练侧是主因 |
| < 0.0986 | < -3% | beam size 不是根因 (GPU 噪声可能) |

---

## 4. 预算

| 阶段 | 估算 |
|------|------|
| Stage 4 re-eval (num_beams=50) | ~5 min |
| **总计** | **~5 min** |

**注**: Task #155 是最便宜的 ablation, 不需要重训. 用 Task #84 best ckpt 直接重跑 Stage 4 即可.

---

## 5. 风险

- **beam 50 推理时间 2.5×**: 20 beam 是 5 min, 50 beam 可能 ~12 min
- **OOM 风险**: GPU 3 47 GB 空闲, batch=96 足够, 无 OOM 风险

---

## 6. 完成度跟踪

- [ ] Stage 4 re-eval (num_beams=50)
- [ ] 写 verdict `verdicts/task155_hgrec_beams50_result.md`
- [ ] 更新 loop.md §16

---

## 7. 用户说明 (R11.3 自主决策)

用户原始提议是 "filter_items=True vs False". HG-Rec pipeline 无此概念 (HG-Rec 是 generative beam search, 不做候选过滤). 我用 `num_beams=20 → 50` 作为最接近的替代对照 (增大搜索空间).

如果用户期望的是 LETTER-style 评估协议 (候选集 + 历史过滤), 那需要新写一个 LETTER-style eval wrapper 套在 HG-Rec ckpt 上, 这是另一个独立任务 (~3h ROI).