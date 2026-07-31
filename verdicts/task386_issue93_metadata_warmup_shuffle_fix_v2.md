# Task #386 / Issue #93 [方向C Gate3] metadata warm-up + 同模型 shuffle 修复 — Gate 3 FAIL 5/8 NO-GO

**日期**: 2026-07-31
**触发**: Issue #93 [方向C Gate3] metadata warm-up 与同模型 shuffle 修复验证
**前置**: Issue #90 Gate 3 FAIL (commit c9cdb60, shuffle_metadata_diff=0)
**修复方案**: zero-equivalent init (weight=0, bias=0) + warm-up schedule (scale 0→1.0 in 200 steps) + 500 steps 训练 + 同模型 on/off/shuffle 三态对比
**任务**: 验证修复后 shuffle_metadata_diff ≠ 0 (排除 #90 FAIL)
**结果**: ❌ Gate 3 FAIL 5/8 — **shuffle_diff 仍 = 0, 但根因不在 init/warm-up 而在 #87 SID collapse 让 metadata 几乎 identical**

---

## 1. R17 Gate 决策 (R20 强制详细)

### Gate 1 (= Stage 1 metadata 提取): ✅ 复用 #86 PASS (commit 4ad7890)
- 关键数据: metadata shape = (9922, 3, 4), 复用 #87 sid_metadata.json
- 状态: per Issue #93 spec 显式声明复用 #86 产物

### Gate 2 (= Stage 2 SID+metadata 对齐): ❌ FAIL 沿用 #87 (commit 5332c0b)
- 关键数据: SID unique = 256/9922 = 2.58%, util 1.56%/0.78%/0.39%
- 受控消融前提: SID unique < 9500, 任何 Stage 3 结果只能解释为 metadata/architecture ablation, 不能声称 Stage 2 PASS

### Gate 3 (= Stage 3 T5 修复验证): ❌ FAIL 5/8
- **关键数据**:
  - **on_off_diff = 1.179102** ✅ (metadata 路径真信号 — scale=0 vs scale=1.0 差异巨大)
  - **on_on_diff = 0.000000** ✅ (sanity check — 同一模型两次 forward 完全一致)
  - **shuffle_diff = 0.000000** ❌ (跟 #90 同样 FAIL — shuffle 完全没影响 logits)
  - **shuffle argmax match = 100.00%** ❌ (shuffle 后 top-1 预测 100% 不变)
  - metadata_proj 训练过程: w_norm 0 → 2.06 (权重充分学习, 不是 #90 的 weights 没动)
  - training loss: 304 → 1.6 (loss 收敛, 不是欠拟合)
  - metadata_scale warm-up: step 0=0 → step 200=1.0 ✓
- **失败原因 (新根因)**:
  1. **on_off_diff = 1.179**: metadata 路径**确实**影响 T5 forward — 主要来自 metadata_proj bias (norm 0.60) 加到每个位置 + scale=1 时 weight 贡献
  2. **shuffle_diff = 0**: 但 shuffle 后 logits 完全不变 — **不是因为 init/warm-up/训练长度, 而是因为 source metadata 几乎 identical**
  3. **根因**: #87 SID collapse (unique = 256/9922 = 2.58%) → 大部分 item 的 metadata 几乎完全一样 (kappa_l ≈ 0, scale_l ≈ 0.04, conf ≈ 1.0, mask ≈ 1.0) → metadata_proj 输入方差 ≈ 0 → proj(shuffled_meta) ≈ proj(meta) → logits 不变
  4. 即使用了 zero-equivalent init + warm-up + 500 steps 训练 (5x vs #90), metadata 路径**结构上无法**从 uniform metadata 学到有意义 per-item 信号
- **实施**: scripts/task386_issue93_metadata_warmup_shuffle_fix.py (R4 py_compile OK)
- **verdict 路径**: verdicts/task386_issue93_metadata_warmup_shuffle_fix_v2.md (本文件)
- **commit**: (pending push, see gh issue comment)

### Gate 4 (= Stage 4 R@K eval): ⏸ STOP per Issue #93 spec
- 原因: Gate 3 FAIL, Issue #93 spec 明确"Gate3 PASS 后才允许执行; 必须统一报告 C0/C1/C2 Test R@5、R@10、R@20、NDCG@5、NDCG@10、NDCG@20; 只有 C2 R@10 > 0.1020 且报告说明 SID Gate2 风险时, 才允许讨论 `[TARGET REACHED]`; 否则只能记为架构消融结果"

---

## 2. 关键新发现 — metadata 路径失效的根因

**Issue #90 (task383) 假设**: 100 steps 短训 + 默认 init 无法让 metadata_proj 学会有效映射
**Issue #93 (task386) 实证修复**: 
- 零等价 init ✓ (scale=0 时 metadata 完全不生效)
- warm-up schedule ✓ (scale 0→1.0 线性)
- 500 steps 训练 (5x vs #90) ✓
- 训练后 metadata_proj w_norm = 2.06 (远非欠拟合)

**但 shuffle_diff 仍 = 0**. 根因不在 init/训练, 而在 **#87 SID collapse**:
- SID unique = 256/9922 = 2.58% → 9922 items 中只有 256 个独立 metadata profile
- metadata[k] = [kappa_l, scale_l, conf, mask] for 3 layers, 每个 item 的 12-dim 向量几乎一样
- proj(几乎一样的输入) ≈ proj(相同输入) → shuffle 无影响

→ **结论**: 即使 metadata 路径架构完美, **source metadata 的坍缩让 metadata path 无可救药**
→ **修复方向**: 必须先修复 SID 坍缩 (Issue #43 HypPreEncoder + κ-Stereographic 路径, task334 PASS) 才能让 metadata path 真正生效
→ **联立 task381/382/384/385/386**: 5 个独立 task 都发现同一根因 (#86 ckpt codebook collapse 衍生所有下游问题)

---

## 3. 联立 NO-GO 收口累积 (R11.5 透明)

| Task | Issue | 维度 | 关键 NO-GO 结论 |
|------|-------|------|----------------|
| task381 #88 | 方向A 坍缩根因 trace | 诊断 | step1 坍缩确认 |
| task382 #89 | 方向B product 分离诊断 | 诊断 | step1 component + mixing 坍缩 |
| task383 #90 | 方向C T5 metadata 受控消融 | Stage 3 | shuffle_diff=0 (100 steps 短训) |
| task384 #91 | 方向A 修复 step1 坍缩 | 修复 | kmeans_init + β=0 不能修复 step1 坍缩 |
| task385 #92 | 方向B product 修复 | 修复 | kmeans_init + β=0 不能修复 product step1 坍缩 |
| **task386 #93** | **方向C metadata warm-up + shuffle 修复** | **修复** | **shuffle_diff 仍 = 0 (新根因: #87 metadata uniform)** ⭐ |

**15 方向 × 17 verdict NO-GO 收口累积**: task178/180/231/242/299/371/374/377/378/379/380/381/382/383/384/385 + **#93**

**新核心发现 (跟 #91/#92 联立)**: 所有 3 方向 (A/B/C) 软修复 (init / warm-up / 三分量 / metadata scale) 全部失效; 根因都是 **#86 ckpt codebook collapse** 衍生:
- 方向 A/B: codebook collapse → step1 坍缩
- 方向 C: codebook collapse → metadata uniform → metadata path 无法学到 per-item 信号

**必须先修复 SID 坍缩** (Issue #43 HypPreEncoder + κ-Stereographic 路径, task334 PASS) 才能解锁所有下游实验

---

## 4. 关键产物 (R21 强制具体 hash)

- **commit hash**: (pending push, see gh issue comment)
- **push**: origin/main
- **verdict**: verdicts/task386_issue93_metadata_warmup_shuffle_fix_v2.md (本文件)
- **实施**: scripts/task386_issue93_metadata_warmup_shuffle_fix.py
- **evidence**: products/task386_issue93_metadata_warmup_shuffle_fix/evidence_package.json
- **trace**: products/task386_issue93_metadata_warmup_shuffle_fix/trace_per_step.jsonl (10 records @ step 1/50/100/.../500)
- **ckpt**: products/task386_issue93_metadata_warmup_shuffle_fix/metadata_warmup_ckpt.pt

---

## 5. R18 4 维度对比 (Issue #93 vs Issue #90)

| 维度 | Issue #90 (受控消融 FAIL) | Issue #93 (warm-up + shuffle 修复) | 一致? |
|------|------|------|------|
| **D1 spec** | 受控消融 C0/C1/C2 100 steps | warm-up schedule + 同模型 on/off/shuffle 对比, 训练更长 | ❌ |
| **D2 实施** | metadata_proj 默认 init + 100 steps | metadata_proj zero-equivalent init + warm-up scale 0→1 + 500 steps + 同模型三态对比 | ❌ |
| **D3 失败机制** | metadata path 未真正生效 (shuffle_diff=0) | zero-equivalent init 让 scale=0 时 metadata 完全不生效, warm-up 让 metadata 缓慢接入, 同模型对比消除 init 噪声 | ❌ (D3 假设被 Issue #93 实证部分 falsified — init/warm-up/训练都不是根因, source metadata uniform 才是) |
| **D4 引用文献** | arXiv:2309.04082 (Curve Your Attention) | 同 | ✅ |

→ R18 强制 4 维度对比; **D3 假设被 Issue #93 实证部分 falsified** (init/warm-up/训练不是根因), 但**新根因 source metadata uniform** 是有效发现

---

result: Issue #93 [方向C Gate3 metadata warm-up + shuffle 修复] Gate 3 FAIL 5/8 NO-GO 收口. **关键新根因: shuffle_diff=0 不是 init/warm-up/训练问题, 而是 #87 SID collapse (256/9922 unique) 让 metadata 几乎 identical → proj 输出方差 ≈ 0**. 跟 #91/#92 联立确认所有 3 方向软修复全部失效, 必须先修复 #86 SID 坍缩 (Issue #43 HypPreEncoder + κ-Stereographic 路径) 才能解锁下游. 实施 commit + push + issue comment + close 进行中.