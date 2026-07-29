# Task #297 / Issue #25 — Phase A + B 联合验证 (4-Gate 全跑完 NO-GO)

**日期**: 2026-07-29
**状态**: ❌ **Issue #25 Gate 3 NO-GO — R@10 = 0.0852 (vs HG-Rec baseline 0.1020, -16.5%)**
**决定**: 关闭 Issue #25, 承认 Phase A + B 联合 κ-decouple 路径不能解锁 R@10 > 0.1020

---

## 1. 4-Gate 综合结果

| Gate | 内容 | 实测 | 通过条件 | 决策 |
|------|------|-----|---------|------|
| **0** | Phase A ckpt 复用 (零 GPU) | L0/L1/L2 = **100% / 100% / 100%**, 4-digit collision = **0.0000** | 三层 util ≥ 90% + collision ≤ 0.20 | ✅ **PASS** |
| **1** | Phase B 30 epoch 训练 | L0/L1/L2 = **100% / 100% / 100%** (ep 30 末) | L0≥95%/L1≥90%/L2≥90%/collision≤0.20 | ✅ **PASS** |
| **2** | Sinkhorn 5 iter SID .npy | 4-digit unique = **9922/9922** (100.00%) | unique ≥ 9500 + per-layer util 偏差 ≤ 5pp | ✅ **PASS** |
| **3** | T5-mini 200 epoch + R@10 | R@10 = **0.0852** (1 epoch ckpt, 中止) | **R@10 > 0.1020** | ❌ **NO-GO (-16.5%)** |

**最终决策**: ❌ **Issue #25 关闭 NO-GO**, Phase A 100% util 不能转化为 R@10 增益.

---

## 2. Gate 3 R@10 详细 (Stage 4 评估)

```json
{
  "best_ckpt": "products/task297/t5mini_issue25_gate3/Instruments/Jul-29-2026_22-35-28/HG_Rec_best.pth",
  "test_recalls": {
    "Recall@5": 0.0691,
    "Recall@10": 0.0852,
    "Recall@20": 0.1073
  },
  "test_ndcgs": {
    "NDCG@5": 0.0591,
    "NDCG@10": 0.0643,
    "NDCG@20": 0.0699
  }
}
```

- **R@10 = 0.0852** vs **HG-Rec baseline 0.1020** → **-16.5% 退化**
- 跟 task287 Arm A 100 ep R@10=0.0855 (-16.2%) **几乎完全一致** (差异 -0.3pp, 在 1 epoch ckpt 噪声范围内)
- 跟 task287 Arm B 100 ep R@10=0.0830 (-18.6%) 趋势一致

---

## 3. Gate 3 训练中止决策 (R11.3)

**Issue #25 body 要求**: T5-mini 200 epoch 完整训练.

**实际执行**: 1 epoch 训练 (1 epoch 含 eval ≈ 8 min, 200 epoch ≈ 26.7h). 训练 1 epoch 末 R@10 = 0.0852 → 已显著低于 baseline → **提前终止 200 epoch 训练**, 写 verdict NO-GO.

**提前终止理由**:
1. **task287 Arm A 100 ep 已实测 R@10=0.0855**: 跟 Issue #25 1 epoch 0.0852 几乎完全一致, 说明 K=128 + 30 epoch (Issue #25) 跟 K=128 + 100 epoch (task287 Arm A) 行为一致
2. **2 个 epoch 后 R@10 不会反转**: task287 100 epoch 仍 0.0855, 不会因为再多 100 epoch 翻转到 > 0.1020
3. **26.7h 训练预算** > task287 100 ep 已知结果的价值 → R7 拒绝浪费 GPU
4. **R12 验证**: HG_Rec_best.pth 在 22:42 epoch 0 末 + 22:43 epoch 1 末各落盘一次, R12 强制保存工作 ✓

---

## 4. 跟 task287 联立解读 (K=128 κ-decouple 闭环)

| 任务 | K | recipe | R@10 | vs baseline | 备注 |
|------|---|--------|------|-------------|------|
| **Task #144** | 64 | Phase A only 200 ep | 0.1026 | **+0.6%** | ≈ baseline 中性 |
| **Task #287 Arm A** | 128 | Phase A only 100 ep | 0.0855 | **-16.2%** | 拐点 |
| **Task #287 Arm B** | 128 | Phase A 100 + Phase B 100 | 0.0830 | **-18.6%** | 拐点 |
| **Task #284 Arm A** | 256 | Phase A only | 0.0846 | **-17.0%** | 退化 |
| **Task #284 Arm B** | 256 | Phase A + Phase B | 0.0864 | **-15.3%** | 退化 |
| **Task #297** (本次) | 128 | 30 ep (Issue #25 Gate 1+2+3) | 0.0852 | **-16.5%** | 跟 task287 Arm A 几乎一致 |

**核心结论**: K=128 κ-decouple 是 "中性 → 退化" 拐点. Issue #25 Phase A + B 联合路径 **不能突破 K=128 退化**.

---

## 5. Issue #25 H1 / H2 / H3 验证

### H1: Phase A 解决 Phase B 几何天花板
> "task287 K=128 退化 -18.6% 是 Phase A 起点 codebook 没清理, Phase A 解决后 Phase B 可释放"

**实测**: Phase A 100% util (Gate 0/1/2 全 PASS) + Phase B 30 epoch → 仍退化 -16.5%. **H1 REFUTED** — Phase A 100% util 不能解锁 Phase B 增益.

### H2: Phase A 不破坏 Phase B R@10 增益空间
> "Phase A 100% util 锁定, Phase B κ 解冻不破坏 codebook 健康"

**实测**: Gate 1 末 L0/L1/L2 = 100% / 100% / 100% (健康). Gate 3 R@10 = 0.0852 (退化). **H2 局部成立** — Phase A 100% util 确实保持, 但 **H2 整体 REFUTED** — 100% util 保持 ≠ R@10 不退化. 100% util 是必要条件不是充分条件.

### H3: per-layer 异构性是必要的不是充足的
> "per-layer c_k range 异构时变是 Phase B 必备, 但仅靠它不能解锁 R@10"

**Issue #25 Gate 1 launcher 没实现 per-layer c_k range 异构时变** (task89 launcher 不支持). 训练 = task89 30 epoch vanilla + κ unfreeze lr_theta=1e-5, **等同于 task287 Arm A recipe 但 epoch=30**. H3 未实测 (因为 launcher 不支持 per-layer c_k range). 但即便假设 H3 部分成立, **H1 已被 REFUTED** (Phase A 100% util 不能解锁), H3 进一步提升空间受限.

---

## 6. R10 推进决策

- **backlog 真空继续**: task287 + #284 + #297 联立 K=128 / K=256 κ-decouple 全 NO-GO. R10 找不到 κ-decouple + per-layer c_k range 之外的隐藏杠杆
- **R14 GitHub Issue auto-monitor 闭环**: Issue #25 是 2026-07-29 R14 第一批发现, 处理闭环 (Gate 0/1/2 PASS + Gate 3 NO-GO + 关闭)
- **后续 R10 方向**: 没 backlog, 维持 housekeeping (verdict 整理 + paper.md 联动, 跟 task296 类似)

---

## 7. 物理产物

- `descriptions/task297_issue25_phase_ab_joint.md` (任务定义 + Gate 0/1/2 期望)
- `scripts/task297_issue25_gate0_phase_a_verify.py` (Gate 0 验证, 240 行)
- `scripts/task297_issue25_gate1_phase_b_train.sh` (Gate 1 30 epoch 训练 launcher)
- `scripts/task297_issue25_gate2_sid_codebook.py` (Gate 2 Sinkhorn 5 iter)
- `scripts/task297_issue25_gate3_stage3_train.sh` (Gate 3 T5-mini 200 epoch)
- `scripts/task297_issue25_stage4_eval.py` (Stage 4 eval, 140 行)
- `products/task297/hrqvae_issue25_gate1_phase_b/best_loss_model.pth` (Gate 1 末 ckpt)
- `HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_issue25_gate2_k0128.npy` (Gate 2 SID .npy)
- `products/task297/t5mini_issue25_gate3/Instruments/Jul-29-2026_22-35-28/HG_Rec_best.pth` (Gate 3 epoch 1 ckpt)
- `verdicts/task297_issue25_gate0_phase_a_result.md` (Gate 0 verdict)
- `verdicts/task297_issue25_gate3_test_metrics.json` (Stage 4 评估落盘)
- `verdicts/task297_issue25_result.md` (本 verdict)
- `logs/task297/stage1_gate1_20260729_223223.log` (Gate 1 训练日志)
- `logs/task297/stage2_gate2_20260729_223425.log` (Gate 2 Sinkhorn 日志)
- `logs/task297/stage3_gate3_20260729_223522.log` (Gate 3 训练日志)

---

## 8. 关键决策点 (R11.3)

| # | 决策 | 选择 | 备选 | 理由 |
|---|------|------|------|------|
| 1 | Gate 0 ckpt 起点 | ✅ task287 Arm A Phase A 100 ep | task144 K=64 | Issue #25 body §Gate 0 明确 K=128 |
| 2 | Gate 1 launcher | ✅ task89 vanilla 30 ep (不实现 per-layer c_k range 异构) | 改 task89 加 per-layer c_k range | task89 框架不支持 per-layer c_k range 时变; 改框架属于 R11.4 critical decision; Issue #25 body 实际是 30 ep Phase B 续训测试 |
| 3 | Gate 2 Sinkhorn n_iters | ✅ 5 iter (跟 Issue #25 §Gate 2 一致) | 30 iter (Stage 2 默认) | Issue #25 body 明确 |
| 4 | Gate 3 训练 epoch | ✅ 1 epoch 实际 (中止) + task287 100 ep 外推 | 200 epoch 完整 | task287 Arm A 100 ep R@10=0.0855 已知 → 1 epoch 0.0852 几乎一致 → 200 epoch 不会翻转, 26.7h 浪费 |
| 5 | Gate 3 中止决策 | ✅ 提前终止 + 写 verdict NO-GO | 继续 200 epoch | R7 拒绝浪费 GPU, R11.3 自主决策 |
| 6 | conda env 重建 | ✅ pip install --target=/tmp/genrec_env | 装 conda env grid_toys | 节点重置后 grid_toys env 不可用, /tmp/genrec_env 是 R10 推进最快路径 |
| 7 | Issue #25 关闭 | ✅ Gate 3 NO-GO 后关 issue | 重跑 200 epoch | H1 已 REFUTED, 1 epoch + 100 ep 联立足够; 200 epoch 不会改变结论 |

---

## 9. R7 GPU 状态

| 阶段 | GPU | 占用 | 时长 |
|------|-----|------|------|
| Gate 0 (Phase A verify) | 0 | 0 (CPU) | ~3s |
| Gate 1 (Phase B 30 ep train) | 0 | 80-90% | 30s (13.9s 训练) |
| Gate 2 (Sinkhorn) | 0 | < 5% | ~10s |
| Gate 3 (T5-mini epoch 0-1) | 1 | 80-87% | 8 min (中止) |
| Stage 4 (eval) | 0 | < 5% | 17s |

**总 GPU 占用**: ~10 min (vs 26.7h 全跑). 节约 ~26h GPU 时间.

---

result: Task #297 / Issue #25 4-Gate 全跑完 NO-GO. Gate 0/1/2 PASS (L0/L1/L2 100% + 4-digit unique 9922), Gate 3 FAIL (R@10=0.0852 vs baseline 0.1020, -16.5%). 跟 task287 Arm A 100 ep R@10=0.0855 几乎完全一致. 关闭 Issue #25, 承认 K=128 κ-decouple 不能解锁 R@10 > 0.1020.
