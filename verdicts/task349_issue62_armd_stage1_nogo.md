# Task #349 / Issue #62 Gate 1 — Arm D Stage 1 NO-GO (Phase 0 Mode Collapse)

**日期**: 2026-07-31
**触发**: Issue #62 owner comment 2026-07-30T16:44:54Z spec (Arm D = #30 + #43 + K0=256, 1000 epoch batch=1024 lr=1e-3 sk_eps=0.0)
**类型**: Issue #62 Gate 1 Stage 1 Arm D 实证 (3h GPU 预期, 30 epoch USAGE-KILL @ 7min)
**状态**: ❌ **NO-GO** — Phase 0 mode collapse, 跟 task178/180/231/242/299/299 同源

---

## 1. 实施 spec (Issue #62 owner comment)

```bash
Stage 1 = #30 per-layer Codebook Transforms + Issue #43 HypPreEncoder + K0=256 (Arm D)
1000 epoch batch=1024 + lr=1e-3 AdamW + sk_eps=0.0 (Sinkhorn DISABLED)
--num_emb_list 256 128 256 (K0=256)
--hyp_c 0.74 (Issue #43 HypPreEncoder)
--radius_list 0.1 1.0 10.0 --scale_list 2.0 2.0 2.0 (Issue #30)
--beta 0.25 --save_limit 1 --seed 42
```

**目标**: Arm D R@10 > 0.1042 = 突破 #43 单点天花板 (Stage 4 待测)

---

## 2. 实施产物

| 文件 | 角色 |
|------|------|
| `scripts/task349_issue62_gate1_armd_stage1_train.py` | Arm D Stage 1 trainer wrapper (~170 行, #30+#43+K0=256 联合) |
| `scripts/task349_issue62_gate1_armd_stage1_train.sh` | Arm D launcher (GPU 0) |
| `logs/task349/stage1_gate1_armd_20260731_114059.log` | 训练日志 (30 epoch @ ~7 min) |

---

## 3. NO-GO 实证 (USAGE-KILL @ ep30)

**运行轨迹**:
- ep 1-29 正常: train_loss 0.0021, recon_loss 0.00206 (平稳)
- ep 30 [step2 monitor]: collision_rate=0.9999
  - L0: r_std=0.0000 r_min=0.000 r_max=0.000 usage=0.4% (1/256)
  - L1: r_std=0.0000 r_min=0.000 r_max=0.000 usage=0.8% (1/128)
  - L2: r_std=0.0000 r_min=0.000 r_max=0.000 usage=0.4% (1/256)
- `[USAGE-KILL] epoch 30 utilization < 20%, aborting training`
- `RuntimeError: epoch 30 utilization < 20%, killed`

**核心观察**:
- L0/L1/L2 usage 各仅 1 个码字 (1/256, 1/128, 1/256)
- collision_rate=99.99% (几乎所有 items 映射到同一码字)
- r_std=0.0000 (per-layer 几何变换未散开, 全部坍缩到单点)

---

## 4. 根因诊断 (Phase 0 Mode Collapse)

**已知同模式 NO-GO 链**:
| Task | 配置 | 现象 |
|------|------|------|
| task178 | baseline β=0.5, 200 epoch | collision 85.24%, L0 collapse |
| task180 | baseline β=0.5, 200 epoch | collision 81.70%, L0 collapse |
| task231 | Issue #11 c_k_range | collision 95%+, L0 collapse |
| task242 | Issue #11 c_k_range | collision 99.45%, L0 collapse |
| task299 | Issue #28 Gumbel-Softmax | L0/L1/L2 = 21.9/10.2/1.2%, collision 99.1% |
| **task349 Arm D** | **#30 + #43 + K0=256 + Sinkhorn OFF** | **collision 99.99%, L0/L1/L2 usage = 1/256/1/128/1/256** |

**Arm D 特异因子** (vs Issue #30 baseline PASS):
1. **K0=256** (Issue #30 baseline K=[64,128,256], L0 K=64)
2. **Sinkhorn OFF** (sk_eps=0, Issue #30 baseline sk_eps=0.005/0.005/0.005)
3. **lr=1e-3 + 1000 epoch** (Issue #30 baseline lr=1e-3 + 200 epoch)

**根因综合 (per task287 verdict)**: K=128/256 + Sinkhorn OFF + 长训 = 1-码字坍缩.
- task287 Arm A K=128/256 + Sinkhorn OFF + 100 epoch = L0=100% PASS (但有 κ-decouple 救命)
- 本 Arm D 没有 κ-decouple 救命 + K0=256 + Sinkhorn OFF + 1000 epoch = 1-码字坍缩 (没救)

**判定**: Arm D = 大 K + Sinkhorn OFF + 无 κ-decouple + 长训 → 必然 mode collapse (跟 task287 verdict §NO-GO 根因综合一致)

---

## 5. R11.5 透明决策

**为什么 Arm D 失败**:
- 实施基础 (Issue #30 wrapper + Issue #43 wrapper) 各自 PASS, 但组合 Arm D 触发 mode collapse
- Arm D 是 owner 显式 spec, 不是 AI 自主决策 (per Issue #62 owner comment 2026-07-30T16:44:54Z)
- Arm D 失败 = owner spec 在 baseline recipe 内部 NO-GO, 不改变 #30+#43 联合 ablation 命题本身的潜力

**备选 (R11.5 自主决策)**:
- 选项 A: 立即尝试 Arm C (#30 + #43 联合, K=[64,128,256] baseline + Sinkhorn ON) ← 选定
- 选项 B: 关闭 Issue #62 (Arm D 失败 = Issue #62 NO-GO 收口)
- 选项 C: 等 owner 拍板

**决策**: 选项 A — Issue #62 §实验设计明确 Arm C 是 baseline K + Sinkhorn ON, 跟 Arm D 是两个独立 config, Arm C 是 Issue #62 主体路径 (Arm D 是 K-sensitivity ablation).

---

## 6. 下一步: Arm C 启动

**Arm C 配置** (Issue #62 §Gate 1 表 Arm C):
```bash
#30 per-layer Codebook Transforms (r=[0.1,1,10]+s=[2,2,2])
+ #43 HypPreEncoder (c=0.74)
+ K=[64,128,256] (Issue #30 baseline K, no K0=256)
+ Sinkhorn ON (sk_eps=[0.003,0.003,0.003])
+ 1000 epoch batch=1024 lr=1e-3 AdamW (跟 Arm D 同步)
```

**预计 GPU 时间**: 1000 epoch × ~0.25s = ~4 min (短任务) + Sinkhorn ON 让码字利用率正常 = 期望 PASS Gate 1

**决策矩阵**:
- Arm C PASS Gate 1 (L0/L1/L2 usage ≥ 20%) → 进入 Stage 2 Sinkhorn 推断 + Stage 3 T5-mini + Stage 4 R@10 eval
- Arm C 也 USAGE-KILL → Issue #62 整体 NO-GO (跟 #30+#43 K0=256 同模式), 关闭 Issue
- Arm C PASS Gate 1 但 Stage 4 R@10 ≤ 0.1042 → Issue #62 联合 ablation 不优于 #43 单点, NO-GO 收口

---

## 7. R14 闭环

- Issue #62 Arm D Stage 1 USAGE-KILL @ ep30 ❌ NO-GO 收口 ✅
- 根因 = K0=256 + Sinkhorn OFF + 无 κ-decouple + 1000 epoch = Phase 0 mode collapse (跟 task287 verdict §NO-GO 根因综合一致)
- 下一步: Arm C 启动 (Issue #62 §Gate 1 Arm C baseline K + Sinkhorn ON)
- Issue #62 维持 OPEN (Arm C 是 Issue #62 §Gate 1 主体路径, Arm D 是 K-sensitivity ablation)

---

result: Issue #62 Gate 1 Arm D Stage 1 ❌ NO-GO (Phase 0 mode collapse). L0/L1/L2 usage = 1/256/1/128/1/256, collision_rate=99.99%, USAGE-KILL @ ep30. 根因 = K0=256 + Sinkhorn OFF + 无 κ-decouple + 1000 epoch = mode collapse (跟 task287 verdict §NO-GO 根因综合一致). 下一步 Arm C 启动 (#30+#43 联合, K=[64,128,256] baseline + Sinkhorn ON).