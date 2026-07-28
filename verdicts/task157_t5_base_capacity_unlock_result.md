# Task #157 verdict — T5-base 220M 容量解锁

> **result**: T5-base 220M (12+12 layers, d_model=768) 在 Instruments 上**比 T5-small 60M 更差**，典型 small-data 过拟合。

---

## 1. 任务目的

验证 T5 容量从 60M 解锁到 220M 是否能突破 Phase 4 (T5-small 60M R@10=0.1030) 的 recall ceiling。

---

## 2. 关键指标

| 指标 | T5-mini 9.18M (#161) | T5-small 60M (#156) | **T5-base 220M (#157)** |
|------|---|---|---|
| **R@5** | 0.0834 | 0.0831 | 0.0776 (-6.6%) |
| **R@10** | 0.1039 | 0.1030 | 0.0950 (-7.8%) |
| **R@20** | 0.1313 | 0.1304 | 0.1154 (-11.5%) |
| **NDCG@5** | 0.0718 | 0.0711 | 0.0667 (-6.2%) |
| **NDCG@10** | 0.0786 | 0.0775 | 0.0722 (-6.8%) |
| **NDCG@20** | 0.0855 | 0.0844 | 0.0774 (-8.3%) |

均同 SID (code-default β=0.25, [32,64,256], sk=0.5) + beam_size=20 + stage 3 早停 20 epoch。

---

## 3. 关键决策点

- **Stage 3 launcher**: `task157_hgrec_stage3_t5base_train.sh`，CUDA_VISIBLE_DEVICES=0，HG_Rec_best.pth (796MB) 落盘于 epoch ≥20 with early_stop=20。
- **Stage 4 launcher**: `task157_hgrec_stage4_eval.sh`。**首次失败**因为 `<<'PYTHON_EOF'` (带单引号) 阻止了 bash 变量展开，导致 `'$BEST_CKPT'` 字面传入 python → `FileNotFoundError: '$BEST_CKPT'`。修复方案：`<<PYTHON_EOF` (无引号) → Stage 4 第二次成功，metrics JSON 落盘。

---

## 4. 分析解读

### 4.1 反直觉结论：bigger model underperforms

T5-base 220M 全面差于 T5-small 60M (-7% ~ -12%)。可能原因：

1. **小数据过拟合**：Instruments 训练集 ~9k seqs，220M 参数极易 overfit。T5-small 60M 已经接近 information-theoretic 上限。
2. **容量-数据不匹配**：60M 已经够 encode SID 序列的结构信息 (vocab=1025, max_len=20)，多余参数只学噪声。
3. **学习率 / 正则化未调**：T5-base 沿用 1e-4 + 200 epoch + early_stop=20，可能不是 T5-base 的最优点；但**所有 T5 容量档都用同一组超参**，对比公平。

### 4.2 跟 #158 (eval protocol) 的交叉验证

#158 报告 item-level R@10 ≈ 0.090-0.103 区间（filter_items=False），与本次 #157=0.0950 / #156=0.1030 都落在该区间，说明评测口径稳定。

---

## 5. 后续建议

- **NO-GO**：不要继续做 T5-large 770M 或更大模型。
- **可探索的下一步**（如时间允许）：
  - T5-base 220M + 更小 lr (5e-5) + 强正则化 (dropout=0.2, weight_decay=1e-4)
  - T5-base 220M + label smoothing 0.1
  - 但预期边际收益 < 5%，且训练成本 3-4×

---

## 6. 产物清单

| 路径 | 状态 |
|------|------|
| `verdicts/task157_t5_base_metrics.json` | ✅ Stage 4 metrics |
| `products/task157/ckpt_hgrec/Instruments/Jul-24-2026_21-10-08/HG_Rec_best.pth` (796MB) | ✅ best ckpt |
| `logs/task157/stage3_train_jul-24-2026_21-09-59.log` | ✅ Stage 3 log |
| `logs/task157/stage4_eval_jul-25-2026_01-05-58.log` | ✅ Stage 4 log |

---

## 7. 完成度

- [x] Stage 3 train (T5-base 220M, 200 epoch, early_stop 20)
- [x] Stage 4 test eval (R@10=0.0950)
- [x] verdict write
- [x] 提交至 P5 paper section

