# Task #162 — Paper-HG-Rec free-curv per-layer κ (用户提议, R11.3 自主推进)

> **任务目的**: 在 paper-faithful HG-Rec (β=0.5, [32,64,256], sk=0) 上跑 **per-layer L0/L1/L2 κ 随机 init + 可学习**, 看是否跟 Task #89 A 臂 (code-default β=0.25) 一样 κ 全跑回 0, 还是 paper-faithful β=0.5 (commit loss 强度 2×) 启动 κ learning.
> **完成日期**: (in progress)
> **状态**: 🟡 待启动 (R11.3 自主决策, 用户 22:50 提议, 利用 GPU 3 空闲)

---

## 1. 背景

Task #89 A 臂 (code-default β=0.25) + Task #152 L1 对照 已经证明:
- per-layer κ 随机 init → 训练后 κ 全 = 0.000000 (精确)
- L1 init ≠ 0 → κ 仍 → 0 (验证 0 是真实偏好, 不是梯度死区)

**Task #162 假设** (用户 2026-07-24 22:50 提议):
- paper-faithful β=0.5 (vs Task #89 β=0.25, 2× commit loss 强度)
- β=0.5 给 commitment loss 更强梯度推动, 可能让 κ 走出 0?
- 验证 β 在 free-curv codebook 训练中是否影响 κ dynamics

**Task #162 vs Task #89 关键差异**:
| 维度 | Task #89 A 臂 | Task #162 |
|------|---------------|-----------|
| β | 0.25 (code-default) | **0.5 (paper Table 6)** |
| codebook | [64, 128, 256] (code-default) | **[32, 64, 256]** (paper Table 6) |
| sk_epsilons | [0.5, 0.5, 0.5] (code-default) | **[0, 0, 0]** (paper Table 6) |
| θ_init | 0.0 (per-layer κ=0) | **per-layer 随机 sample** (用户在 prompt 强调) |
| seed | 42 | 42 |
| M | 1 (单流形 per layer) | 1 |

其他保持一致.

---

## 2. 实验设计

**Stage 1 (RQ-VAE 训练, GPU 3)**:
- 复用 `scripts/task89_stage1_train_rqvae.py` fork
- args: `--M 1 --beta 0.5 --num_emb_list 32 64 256 --sk_epsilons 0.0 0.0 0.000 --theta_init_list <seed42-random>`
- epochs 1000 (跟 Task #89 一致)
- 输出: `products/task162/rqvae_free_curv/.../best_ckpt.pth` + κ_history.json

**Stage 2 (SID inference, CPU)**:
- 复用 Task #156 rkmeans_inference_flat 模式 (因为 SID inference 只读码本, 跟训练算法无关)
- 输出: `products/task162/Instruments/Instruments_t5_rqvae_paper_free_curv.npy` (N, 4) [追加 1 列去重 digit]

**Stage 3 (T5 训练, GPU 1, 等 #161 完成释放)**:
- 复用 #161 7.6M d_kv fix T5 mini (4 enc + 4 dec layers, d_model=256, 4 heads × d_kv=64)
- Stage 4 code_path 指向 Stage 2 产出的 `_paper_free_curv.npy`
- 估算 ~36 min 早停

**Stage 4 (R@10 eval, GPU 1)**:
- 复用 #161 stage4 launcher (改 code_path → paper_free_curv)
- 输出: `verdicts/task162_*_metrics.json`

**保持不变**: Stage 1 sentence-t5-base embedding (768-dim), data, seed=42

---

## 3. 决策触发 (vs Task #89 A 臂 κ_history)

| κ_history 终值 | 解读 | 决策 |
|---------------|------|------|
| **所有 κ ≠ 0 (学出非零曲率)** | ✅ **β=0.5 启动 κ learning** | paper HG-Rec free-curv 优于 code-default free-curv. 走下游 R@10 评估 + 写 paper section 主张 "paper β=0.5 是 κ 学习关键". |
| **所有 κ → 0** (跟 Task #89 一致) | ❌ **β 不是 κ learning 关键** | 跟 Task #89 NO-GO 一致强化. 写 verdict 归因. |
| **部分 κ ≠ 0, 部分 → 0** | ⚠️ **β 部分影响** | 报告哪个 layer 学出非零, 哪个仍 0. 看下游 R@10. |

下游 R@10 对比:
- Task #89 A 臂 R@10=0.1015 vs baseline 0.1020 (-0.5%)
- Task #162 期望 ≈ 0.102 (在 noise 内) 如果 κ 全 0
- Task #162 如果 R@10 > 0.105 → 真正 free-curv gain

---

## 4. 预算

| 阶段 | 估算时间 | GPU |
|------|---------|-----|
| Stage 1 RQ-VAE (1000 ep) | ~30-40 min | GPU 3 (空闲) |
| Stage 2 SID inference | ~5 min | CPU |
| Stage 3 T5 训练 (7.6M d_kv fix) | ~36 min 早停 | GPU 1 (等 #161 完成) |
| Stage 4 eval | ~1 min | GPU 1 |
| **总计** | **~75-85 min** | (其中 GPU 1 串行 #161 → #162) |

---

## 5. 风险与缓解

**风险 1**: Stage 1 1000 epoch + per-layer κ random init 可能不稳定 (Task #89 θ_init=0 稳定). → **缓解**: 先 200 epoch 试跑看 κ dynamics, 稳定再续 800 epoch.

**风险 2**: GPU 1 等 #161 完成需要 ~30-50 min, 期间 Stage 3 不能启动. → **缓解**: Stage 1 + Stage 2 提前跑 (CPU/GPU 3), Stage 3 等 #161 自然腾出.

**风险 3**: paper-faithful β=0.5 + codebook [32,64,256] 比 code-default [64,128,256] 小一半, codebook 利用率可能下降. → **缓解**: 看 unique SID 数, 跟 Task #156 / #89 对比.

---

## 6. R11.3 决策明示 (写入 loop.md §16 备注)

**(a) 选了哪个**: Task #162 paper-faithful HG-Rec free-curv (β=0.5, [32,64,256], sk=0, per-layer κ 随机 init + learnable)

**(b) 为什么**:
- 用户 2026-07-24 22:50 明确提议 "论文版本的HG-REC + 三层 k 随机初始 + 可学习"
- 跟 Task #89 A 臂 (code-default β=0.25) 不同在 β + codebook + sk
- 历史 Task #89 + #152 NO-GO 已强, 但 paper β=0.5 是 new variable → 值得 sanity check
- 利用 GPU 3 空闲 (R7 + R10 主动推进)

**(c) 备选方案**:
- **A**: 跑 Task #89 A 臂重做 (重复, 不建议)
- **B**: 跑 Task #152 L1 init ≠ 0 加重做 (重复, 不建议)
- **C**: 等 #157/#160 完成 (低 ROI 等待)
- **D**: 跑 Task #162 paper-faithful free-curv (本次决策)

---

## 7. 完成度跟踪

- [ ] Stage 1 RQ-VAE launcher fork + description
- [ ] bash -n 验证 Stage 1 launcher
- [ ] Stage 1 启动 (GPU 3, 监控 κ_history 每 100 epoch)
- [ ] Stage 1 完成 (1000 epoch + best ckpt + κ_history 落盘)
- [ ] Stage 2 SID inference (CPU, ~5 min)
- [ ] Stage 3 T5 launcher fork (复用 #161 + 改 code_path)
- [ ] Stage 3 启动 (GPU 1, 等 #161 完成)
- [ ] Stage 4 eval 完成 + JSON 落盘
- [ ] 比较 κ_history 终值 vs Task #89 + 比较 R@10 vs Task #89/#156 baseline
- [ ] 写 verdict: `verdicts/task162_paper_free_curv_result.md`
- [ ] 更新 §16 R8 cleanup

---

result: Task #162 — Stage 1+2+3+4 全 pipeline 启动 + 完成 + 写 verdict. 重点看 κ_history 终值 vs Task #89 (paper β=0.5 是否能启动 κ learning).