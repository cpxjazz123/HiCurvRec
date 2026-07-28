# Task #202 Sinkhorn-on Stage 3 K0=64 — verdict

> **完成日期**: 2026-07-26 01:38 (early stop, ep 117/200, counter 20)
> **状态**: ❌ **NO-GO — Sinkhorn 路径无增益**

---

## 1. 任务目标 (用户 2026-07-25 拍板)

用户拍板: "用 Sinkhorn 版 SID 重跑 Stage 3 K0=64, 只需要一个".
目的: 验证 Sinkhorn 路径 vs 默认 SID 在 val 上的真实差异 (用户预测 "+1~3%").

启动命令: `bash scripts/task202_stage3_sinkhorn_only.sh`
配置: T5-mini (9.18M), K0=64, Sinkhorn SID `_t5_rqvae_k064_sk0.003.npy`, early_stop=20, 200 epoch, GPU 1.

---

## 2. 关键指标 (best epoch)

| 指标 | **#202 Sinkhorn** | **#181 默认 SID** | #188 K0=64 mean |
|------|---------|---------|---------|
| val Recall@5 | **0.0909** | 0.0899 | — |
| **val Recall@10** | **0.1065** | 0.1057 | 0.1241 (mean of 4 tier × 3 seed) |
| val Recall@20 | 0.1223 | 0.1215 | — |
| val NDCG@5 | 0.0785 | — | — |
| val NDCG@10 | 0.0836 | — | — |
| **val NDCG@20 (best)** | **0.0876** | — | — |

**对比 #181 (Phase 0.6 默认 SID, val R@10=0.1057)**:
- #202 Sinkhorn val R@10 = 0.1065 → Δ = **+0.0008 (+0.07%)** → 持平
- 用户预测 "+1~3%" → **完全没达到** (差 12×)
- Stage 3 训练时长: 80 min (vs #181 类似)

---

## 3. 结论

**Sinkhorn 路径在 Stage 3 上 NO-GO**:
- Sinkhorn SID (sk_eps=0.003) vs 默认 SID (sk_eps=0) 在 Stage 3 val R@10 上 **持平** (差异 < 0.001)
- 用户预测 "+1~3%" 已被实验坐实为 FAIL
- **机制**: Sinkhorn 强制均匀分配 → Stage 1 碰撞率 0%, 但 Stage 2 SID 质量**没有质变**, Stage 3 学到的语义也**没变好**

---

## 4. 产物清单

| 路径 | 大小 | 内容 |
|------|------|------|
| `products/task202/t5mini_k064/Instruments/Jul-26-2026_00-16-30/HG_Rec_best.pth` | 22 MB | R12 best ckpt (epoch 113, val NDCG@20=0.0876) |
| `logs/task202/stage3_k064.log` | 226 行 | 训练 summary log |
| `logs/task202/Instruments/Jul-26-2026_00-16-30/HG_Rec.log` | 完整 | 详细 epoch metrics log |

---

## 5. 后续建议

- ❌ **Sinkhorn 路径不进 paper recipe** (用户拍板的 HG-Rec baseline R@10=0.1020 仍然成立, Sinkhorn 不贡献增益)
- ✅ **HG-Rec paper recipe** (sk_eps=0 + argmin + K0=64) 验证为 baseline — Task #188 paper Table 7 复现已涵盖
- ❌ **不要在 verdict/论文里把 Sinkhorn 列为可调旋钮** — 实验证明它**没影响**

---

## 6. 关键决策点

| 时间 | 决策 | 理由 |
|------|------|------|
| 2026-07-26 00:16 | **启动 #202 Sinkhorn Stage 3** | 用户拍板"用 Sinkhorn 版 SID 重跑 Stage 3" |
| 2026-07-26 01:24 | **最佳 epoch 保存** | val NDCG@20=0.0876 (epoch 113), R12 ckpt 落盘 |
| 2026-07-26 01:38 | **Early stop 触发** | counter 20 (连续 20 epoch val NDCG@20 不改善) |
| 2026-07-26 01:38 | **写本 verdict** | Sinkhorn 路径 NO-GO |

---

## 7. 状态总结

- ❌ **Sinkhorn 路径 NO-GO** — val R@10=0.1065 vs #181 0.1057 持平 (+0.07%, 完全没达到用户预测的 +1~3%)
- ✅ **HG-Rec baseline (sk_eps=0) 维持** — Sinkhorn 不贡献增益, 不进 paper recipe
- ✅ **Stage 3 训练流程验证** — early_stop + best_metric save (R12) 工作正常