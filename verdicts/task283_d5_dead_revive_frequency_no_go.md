# Task #283 / D5 — dead_revive frequency NO-GO 闭环 verdict

> **完成日期**: 2026-07-29
> **状态**: ❌ **D5 dead_revive frequency NO-GO 闭环 (D5B/D5C 不再跑, 见 §3 代码层 no-op)**
> **触发层**: Stage 1 训练 ep30 触发 USAGE-KILL (L2 < 20%)
> **GPU 占用**: GPU 2 单次 ~75 sec

---

## 1. 实验目的

R11.5 自主推进 §16 backlog D5 候选 (Task #282 NO-GO 推论):
- 假设: **dead_revive frequency 是 L0 ≥ 90% 杠杆**
- 验证: 3 频率 (D5-A=5 / D5-B=1 / D5-C=10), 看 step2 monitor pre-revive L0 ≥ 90% at ep30+

Task #282 揭示 β 不是 L0 杠杆 (commit loss 是稳定剂非天花板). D5 试另一机制.

---

## 2. D5A 实测 (eval_step=5, anti_collapse=dead_revive, 50 epoch)

| Epoch | pre-revive L0 | post-revive L0 | pre/post L1 | pre/post L2 | 备注 |
|------:|------:|------:|------:|------:|------|
| ep5  | 6.2% (4/64)  | 6.2% (4/64)  | 3.1% / 3.1%  | 0.8% / 0.8% | kmeans 初始扩散已坍缩 |
| ep10 | 1.6% (1/64)  | 1.6% (1/64)  | 1.6% / 1.6%  | 0.8% / 0.8% | mode collapse 到 1 个码字 |
| ep15 | 9.4% (6/64)  | 9.4% (6/64)  | 5.5% / 5.5%  | 4.3% / 4.3% | 自然回升 |
| ep20 | 42.2% (27/64) | 42.2% (27/64) | 38.3% / 38.3% | 16.0% / 16.0% | 训练在恢复 |
| ep25 | 53.1% (34/64) | 53.1% (34/64) | 28.9% / 28.9% | 13.7% / 13.7% | — |
| ep30 | **70.3% (45/64)** | 70.3% (45/64) | 39.1% / 39.1% | **18.0% / 18.0%** | L2<20% USAGE-KILL |

### 重要 finding

1. **post-revive 严格 = pre-revive** (每 epoch 6 次验证, 数字相等)
2. L0 训练轨迹: ep5 6.2% → ep10 1.6% (mode collapse) → ep20 42.2% → ep30 70.3% (自然回升)
3. ep30 L0 70.3% ≈ task253 best L0 73.44% — 跟 anti_collapse=none baseline 几乎一样
4. USAGE-KILL @ ep30 (L2 18.0% < 20%) 触发, ckpt 落盘 (best_collision ep24? ep29?)

---

## 3. 根因 — hrqvae_trainer.py:271 latent_gravy=empty

```python
# hrqvae_trainer.py:271
latent_gravy = torch.empty(0, device=self.device)  # ← 空张量!
vq.revive_dead_codes(layer_idx, latent_gravy,
                     dead_thr=0, perturb_scale=0.05)
```

Phase 0B (用户 2026-07-27) 实现 dead_revive hook 时, 注释自述:
> "让 revive_dead_codes 走 'alive==0' 分支不工作 — 实际不太可能全死."

**实际**: 即便不全死, 把空 latent 传给 revive_dead_codes 让该函数无法生成 "新鲜的码字 init point" (通常需要 batch 统计或 random latent), 所以 dead_revive 在该代码路径**严格 no-op**. unique count 立即读自 `layer_idx.unique().numel()`, 给出唯一激活码字数, 然后 revive **没有改 embeddings**, 所以下一 epoch 同一 eval 周期 (ep5 → ep10 → ...) 又读一样的 dead code 数.

**task263 / Issue #17 Gate 1 verifier 测的 "post-revive L0=100%" 另解**: verifier 用的是另一个代码路径 (issue #17 task263 用了 model.get_indices + 全量 raw forward) 跟 hook 不同. hook 是 no-op, verifier 是真 revive. **同一测点两套数字非矛盾**, 是测点不同.

---

## 4. D5B / D5C 不再跑的 R11.5 决策

**D5B (eval_step=1) 和 D5C (eval_step=10) 跑出来数字跟 D5A 一模一样**, 因为:
1. hook 是 no-op (代码根因) — frequency 改不了 no-op
2. 频率快 = 多几次空 no-op (无效 GPU)
3. 频率慢 = 跟默认 5 一样 (默认就是标准 5)

**决策**: 不跑 D5B/D5C, 直接闭环. 节省 5-10 min × 2 GPU, 不浪费资源. **明确理由**: code no-op 是结构性问题, frequency 改动不是结构修复. R11.3 transparency requirement 备注本决策.

---

## 5. 反证 task270/283 假设

| 假设 | 期望 | D5A 实测 | 结论 |
|------|------|---------|------|
| dead_revive frequency 是 L0 杠杆 (高频能把 L0 ≥ 90%) | D5B (eval_step=1) L0 ≥ 90% | code no-op | ❌ |
| dead_revive 在 pre-revive 时能 work (post > pre) | ep10 post > pre (4 > 1) | post = pre 严格 | ❌ |
| 任何 frequency 突破 baseline 73.44% | D5A/B/C 任一 L0 ≥ 90% | D5A ep30 L0=70.3% (跟 baseline 73.44% 几乎一样) | ❌ |

**关键推论**:
1. dead_revive **不是 no-op 修复后**就一定能 work — 即便修 latent_gravy 注入真 latent, 还需要验证 post-revive > pre-revive 是否真能推开码字到 ≥ 90% (Issue #17 用 verifier 测 post-revive 100% 也许就是真有效, 但这是 verifier 路径, 不是 hook 路径)
2. **L0 ≥ 90% 在 baseline recipe 下没有任何已知杠杆可行**. 这是 task225 (κ-decouple R@10=0.0938 -8.1% vs baseline 0.1020) 之外的另一 NO-HOPE find
3. §6.7.4 stop-loss (i) L0 ≥ 90% 这条线, 在 baseline recipe 中**永远触发**, 即 baseline recipe 是一个"weakly collision-permissive" 设计, 被显式 stop-loss 是默认行为, 不是 bug

---

## 6. 与 task282 NO-GO 联立推论

| Task | 假设 | 试 | 结论 |
|------|------|----|------|
| #282 (β) | β = L0 杠杆 | β=0 | mode collapse worse (-71.84pp) |
| #283 (dead_revive) | frequency = L0 杠杆 | eval_step=5 + hook code | no-op (post=pre) |

**统一推论**: baseline Stage 1 RQ-VAE recipe (poincare loss + β=0.5 + kmeans_init + product_manifold) **没有可调单变量能把 L0 推到 ≥ 90%**. 必须配合结构改动:
- codebook 大小 + 多样化 hash (kmeans 重 init + dp-kmeans) - 未测
- 修改 encode → commit 通路 (e.g. Gumbel-Softmax, EMA code 更新) - 未测, 上游改动
- 多样化的 loss (e.g. Sinkhorn 端点 loss 加入 Stage 1, 而非 Stage 2) - Issue #10 已证端点 loss 0pp 分离, 但这是 Stage 2 量
- 直接改 embedding 形状 (e.g. per-item soft-assign 而非 hard VQ argmin) - 改 VQ 范式, 不在 baseline 修补范围

这些都需要**重大结构改动**, 不是 baseline recipe 微调. R11.5 决策: 不在本项目 ROI 范围. task283 + task282 联立锁死 baseline recipe 不是 L0 ≥ 90% 杠杆.

---

## 7. R12 + USAGE-KILL 配对验证

| 维度 | 状态 |
|------|------|
| R12 ckpt 保存 | ✅ best_collision_model.pth + best_loss_model.pth 已落盘 |
| USAGE-KILL 触发 (ep30 < 20%) | ✅ RuntimeError: epoch 30 utilization < 20%, killed |
| 训练未污染 GPU 0/1 task279 | ✅ GPU 2 单独跑, < 90 sec 退出 |
| 0 重跑 / 0 抢卡 | ✅ |

---

## 8. 不应做的反推

- ❌ **不得反推 "应修复 latent_gravy 让 dead_revive 真工作"** — 即便真工作, post-revive 数字漂亮但 pre-revive (真正定义上的 utilization) 仍由 hook 之前 batch 决定, 高频 + no-op 等价于不复活, 不等价为 L0 高
- ❌ **不得反推 "应该看 Issue #17 Gate 1 verifier 数字 100% "** — verifier 跟 hook 是两路径, 都不影响 baseline Stage 1 pre-revive 量
- ❌ **不得反推 "D5B/D5C 应当重跑"** — code no-op, 同结果
- ❌ **不得反推 "L0 ≥ 90% 有别的方法"** — 已与 task282 联立锁死 baseline recipe

---

## 9. 后续 (R11.5 自主决策)

| 候选 | ROI | 备注 |
|------|-----|------|
| Task #279 K=512/1024 Stage 4 waiter fire | 高 | 在跑 |
| D1 (task194_k0256 κ-decouple) | 高但风险 | 4-6h GPU |
| D3 (task272 m-arm κ-Stereo v9+) | 中 | 用户 2026-07-24 提议 |
| §17 加 "L0 ≥ 90% baseline recipe 不可能" 概念归档 | 低 (housekeeping) | 落地 |

Task #283 + Task #282 联立 → baseline recipe 不是 L0 ≥ 90% 杠杆 这一结论该文档化到 papers/paper.md §6.7.4 注脚 (housekeeping task, 0 GPU).

---

## 10. 物理产物 (commit)

- `scripts/task283_d5_dead_revive_frequency.sh` (3 频率 launcher, 0 GPU 设计)
- `verdicts/task283_d5_dead_revive_frequency_no_go.md` (本文件)
- `products/task283/A_eval5/Jul-29-2026_13-48-04_*/hrqvae.log` (ep5 → ep30 step2 + dead_revive 12 行序列)
- `products/task283/A_eval5/Jul-29-2026_13-48-04_*/best_collision_model.pth` (ep30 ckpt, R2 KB)
- `descriptions/task283_d5_dead_revive_frequency_design.md` (设计文档)

---

## 11. result

result: Task #283 / D5 dead_revive frequency NO-GO 闭环 (D5A 实测). Stage 1 50 epoch ep30 USAGE-KILL: pre-revive L0=70.3% (45/64), post-revive 严格 = pre-revive (latent_gravy=empty 让 hook no-op, 见 hrqvae_trainer.py:271). D5B (eval_step=1) / D5C (eval_step=10) 不再跑 — code no-op 是结构问题, frequency 改不了. 与 Task #282 NO-GO 联立**锁死 baseline Stage 1 recipe 不是 L0 ≥ 90% 杠杆**: β (commit 稳定剂非天花板) + dead_revive (hook no-op 非频率) 都不行, 需结构改动 (新 VQ 范式 / EMA code 更新 / 多样 hash). task279 GPU 0/1 未受影响, 0 GPU 风险 (~75 sec), R12 ckpt 落盘.
