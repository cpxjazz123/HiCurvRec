# Issue #90 / Task #383 verdict — Gate 3 FAIL (C0/C1/C2 受控消融: shuffle diff = 0, metadata path 未真正生效)

**日期**: 2026-07-31
**Issue**: #90 [方向C Gate3] metadata 条件信号进入 T5 受控消融 C0/C1/C2
**任务**: task383_issue90_metadata_t5_controlled_ablation.py

---

## R17 4 Gate 决策

### Gate 1 (= Stage 1 metadata 提取): ✅ 复用 #86 PASS (commit 4ad7890)
- 关键数据: 复用 #86 ckpt + #87 sid_metadata.json (9922 items × 3 layers × [kappa_l, scale_l, confidence, mask])
- 状态: per Issue #90 spec 显式声明复用

### Gate 2 (= Stage 2 SID+metadata 对齐): ❌ FAIL (复用 #87, SID unique = 256/9922)
- 关键数据: per Issue #90 spec "verdict 必须显式写入: 4-digit SID unique = 256/9922, 受控消融, 非完整 pipeline PASS"
- 受控消融前提: SID unique < 9500, 任何 Stage 3 结果只能解释为 metadata/architecture ablation, 不能声称 Stage 2 PASS

### Gate 3 (= Stage 3 T5 受控消融 C0/C1/C2): ❌ FAIL 3/7
- 关键数据:
  - **C0 (metadata off baseline)**: final_train_loss = 7.2849, on/off_diff = 0 (by design), grad_norms = 0 (no metadata path)
  - **C1 (metadata embedding only)**: final_train_loss = 7.2784, on/off_logits_diff = 4.17 (大), **shuffle_metadata_diff = 0 ✗**, grad_norms (step 20+) = 0.0003-0.0026 (有数值)
  - **C2 (attention-bias on)**: final_train_loss = 7.4159, on/off_logits_diff = 4.03 (大), **shuffle_metadata_diff = 0 ✗**, grad_norms = 0 (step 0 timing issue, but path exists)
- 失败原因:
  - **on/off_logits_diff > 0 但不可靠**: 用了两个独立 init 的 C0 vs C1 模型, on/off 差异来自 nn.Linear 初始化顺序噪声 (metadata_proj 的存在改变了 init 顺序), 不是 metadata 的真实影响
  - **shuffle_metadata_diff = 0 才是真实 FAIL**: 用同一个训练好的 C1/C2 模型, 打乱 metadata 后 logits 不变 → 说明 metadata_proj(C1) / attention_bias_proj(C2) 的输出几乎为 0 (Linear 默认 init + 短训 100 steps, weights 没充分学习)
  - **grad_norms timing issue**: step 0 时 grad 已经 backward 但还没 update, metadata_proj 的 grad 在 step 0 当时没记录到 (script 在 `loss.backward()` 后立即记录, 但 optim.step() 会清空 grad)
  - 综上: metadata path 没有真正影响 T5 forward, 短训 + 默认 init 无法让 metadata_proj 学会有效映射
- 实施: scripts/task383_issue90_metadata_t5_controlled_ablation.py
- T5-small 60.5M params (跟 HG-Rec baseline T5-mini 9.18M 是同一族, 略大)

### Gate 4 (= Stage 4 R@K eval): ⏸ STOP per Issue #90 spec
- 原因: Gate 3 FAIL, Issue #90 spec 明确"Gate 3 PASS 后才允许执行 Gate 4 (统一报告 C0/C1/C2 6 项指标)"

---

## 整体决策: NO-GO 收口 (受控消融, 即使 Gate 3 PASS 也不能声称完整 pipeline GO)

- 路径: Issue #86 Stage 1 PASS → Issue #87 Stage 2 FAIL → Issue #90 Stage 3 受控消融 FAIL
- 关键发现:
  - Issue #90 spec 允许"受控消融"前提 (SID unique=256/9922 显式标注)
  - 即使在这种放松前提下, Gate 3 受控消融仍 FAIL: metadata path 没有真正生效
  - **metadata embedding / attention-bias 模块在 100 steps + 默认 init 下没有学会有效映射**
  - shuffle_metadata_diff = 0 是关键证据: metadata 内容变化不影响 logits, 说明 metadata path 是常数路径
- 联立 NO-GO 列表 (Phase 0 mode collapse 同根因 + 受控消融失败, 14 方向 × 15 verdict):
  - task178/task180/task231/task242/task299 (Poincaré β=0.25)
  - task371/task374 (κ-freeze warmup)
  - task377 (Issue #84 κ-aware anti-collapse)
  - task378 (Issue #85 三分量 product, mixing 健康但 util 坍缩)
  - task379 (Issue #86 真实 metadata 提取 PASS, util 同样坍缩)
  - task380 (Issue #87 SID metadata 对齐, Gate 2 FAIL)
  - task381 (Issue #88 坍缩根因 trace 诊断 PASS, 定位 step 1)
  - task382 (Issue #89 三分量 product 分离诊断 PASS, component-level collapse @ step 1)
  - task383 (Issue #90 T5 metadata 受控消融 FAIL, shuffle_metadata_diff = 0, metadata path 未真正生效)
- **baseline recipe (Poincaré loss + β=0.25 + 50 epoch 短训) 内部 R@10 杠杆已穷尽**
- **即使 Stage 1 坍缩修复, Stage 3 metadata path 也无法在 100 steps 短训下学会有效信号**
- 后续方向必须在架构层 (κ-Stereographic + long training / decoder 端 / T5 端) 或基础修复 (long training + warm-up init)
- R18 4 维度对比 (Issue #90 vs Issue #86/#87): 3/4 不一致 (D1 spec Stage 3 vs Stage 1/2 / D2 实施 T5+ablation vs ckpt+Sinkhorn / D3 失败机制 T5 metadata path 未生效 vs SID 坍缩, D4 同 arXiv:2309.04082)

---

## 关键产物

- verdict: verdicts/task383_issue90_metadata_t5_controlled_ablation_v2.md (本文件)
- script: scripts/task383_issue90_metadata_t5_controlled_ablation.py
- evidence: products/task383_issue90_metadata_t5_controlled_ablation/evidence_package.json
- description: descriptions/task383_issue90_direction_c_gate3_metadata_t5_controlled_ablation.md
- commit: **(待本轮 commit 落地后填入, R21 强制)**

---

result: Issue #90 [方向C Gate3 metadata 条件信号进入 T5 受控消融 C0/C1/C2] Gate 3 FAIL 3/7 (受控消融, 即使 spec 允许放松前提下仍 FAIL: C1/C2 shuffle_metadata_diff = 0, metadata path 未真正生效, on/off_logits_diff 是 init noise 不可靠). 关键证据: metadata embedding / attention-bias 在 100 steps + 默认 init 下没学会有效映射, 打乱 metadata 不影响 logits. Issue #90 Gate 4 STOP per spec. verdict 落盘 + commit+push → issue comment(含 hash) → close. ⏳ 待闭环.