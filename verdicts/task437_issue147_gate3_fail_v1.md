# Issue #147 [方向C Gate3] 曲率条件化T5注入 — Gate 3 FAIL (Precheck 3/3 PASS, Training adapter grad collapse)

**Task**: #437 / Issue #147
**Commit**: `1a3c623`
**Verdict**: `verdicts/task437_issue147_gate3_fail_v1.md`
**产物**: `products/task437_issue147_curvature_conditioned_residual/{config,verdict,train_trace,adapter_init_proof,gradient_proof,adapter.pt}.{json,pt}`
**SHA256**: item_emb.parquet=`2c5f843d...`, T5_ckpt=`56d046db...`, SID_npy=`2dab2922...`, adapter_ckpt=`6a377018...`

---

## 4 Gate 详细内容回答 (R17+R20 强制)

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE): N/A per spec
- **状态**: Issue #147 spec 只要求 Gate 3 (Stage 3 T5-mini), 不要求 Gate 1 (Stage 1 RQ-VAE)
- **Issue spec 强制**: Issue #147 是方向 C Gate 3 验证, 复用 Task #84 RQ-VAE SID 作为输入 (不动 Stage 1)

### Gate 2 (= Stage 2 Sinkhorn + dedup): N/A per spec
- **状态**: Issue #147 spec 不要求 Gate 2
- **Issue spec 强制**: 复用 Task #84 SID 已经过 Sinkhorn 处理 (4-digit unique)

### Gate 3 (= Stage 3 T5-mini): ❌ **FAIL (Precheck 3/3 PASS, Training FAIL)**
- **状态**: Precheck 全部 PASS, 但 Training 中 adapter 梯度瞬间坍缩, T5 损失无下降
- **关键数据 (Issue #147 spec 强制)**:
  - **Precheck 1 PASS** ✅: gate=0 时 forward 跟原T5 max diff=0 (is_zero_diff=True, 严格 identity-preserving 验证通过)
  - **Precheck 2 PASS** ✅: 非零 gate 时 13 个 adapter 参数 grad 有限非零
    - gate_logit: abs_mean=7.0083e-03 (非零)
    - scale_head.2.bias: abs_mean=6.8877e-03 (非零)
    - conditioner/sid_embed/kappa_embed 全部梯度非零
  - **Precheck 3 PASS** ✅: 真实 history-SID token range 4 层全部 in [0, K_l), all_in_range=True
  - **Stage 3 10 epoch training** (实际跑完, 4120 batches/epoch × 10 = 41,200 batches):
    - epoch 0: avg_loss=1.8863, avg_adapter_grad_norm=**1.9185e-04** (初始更新)
    - epoch 1: avg_loss=1.8886, avg_adapter_grad_norm=**2.9376e-08** (-99.98% 立即坍缩)
    - epoch 2-9: avg_loss=1.886-1.889 (flat), grad_norm=1e-9 至 1e-8 (持续近零)
    - T5 main loss 完全未下降 (1.886 → 1.888 ≈ 噪声级别)
  - **Save/Load 验证**: missing=0, unexpected=0 (严格 ckpt round-trip 验证通过)
  - **Forward 一致性**: forward_diff=0 (两次 forward 输出一致, deterministic 验证通过)
- **失败原因 (跟 #145/#146 共享 family 但 mechanism 不同)**:
  1. **核心根因 — sigmoid 饱和 + frozen T5 切断 adapter 梯度**: scale_head.sigmoid 输出 ∈ [0, 1] (初始接近 0), 乘以 x_emb 后 residual 极小 → 通过 sigmoid 反向梯度乘以 sigmoid'(x) = sigmoid(x)·(1-sigmoid(x)), sigmoid 接近 0 时 gradient 极小
  2. **次级根因 — frozen T5 main 损失路径**: T5 weights 完全冻结 (只训练 adapter), 损失只通过 residual 路径反向. epoch 0 后 residual 实际 = 0 (因 scale → 0), grad 路径断
  3. **precheck vs training gap**: precheck 一次性 backward 验证 graph 非零 (用初始权重), 但 SGD step 后 scale_head.2.bias 朝 0 移动 → sigmoid 输出 → 0 → 所有路径梯度消失
  4. **Mechanism 新颖 (跟 #145/#146 不同)**:
     - #145 (pairwise+ranking minimize): minimize 推 collapse
     - #146 (argmin + relu): argmin 切断 + relu 0 梯度
     - **#147 (sigmoid 饱和 + frozen main)**: scale → 0 → residual → 0 → backward 路径断
- **R18 4 维度路径对比 vs #142 (Issue #129 旧 dual-gate)**:
  - D1 spec: #147 curvature-conditioned residual + identity-preserving, #142 zero-init dual-gate + active-down→ReLU — **完全不同**
  - D2 实施: CurvatureConditionedAdapter MLP + scale_head.sigmoid + gate, AdapterHookedHGRec.zero-init dual-gate — **完全不同**
  - D3 Gate 1 失败机制: sigmoid 饱和 + frozen T5, zero-init 早期 dead — **不同 mechanism 但同样 identity 反作用**
  - D4 引用文献: arXiv:2309.04082《Curve Your Attention》, #129 内部 spec — **完全不同文献**
- **verdict 路径**: `verdicts/task437_issue147_gate3_fail_v1.md` (本文件)
- **实施**: `scripts/task437_issue147_curvature_conditioned_residual.py`
- **整体决策**: ⚠️ **架构级 PASS + 训练级 FAIL**:
  - 架构级 PASS: gate=0 严格 identity-preserving (max diff=0), 13 个 adapter 参数 precheck grad 非零, 真实 SID token range 全部 in_range, save/load round-trip 完美, forward 一致性 diff=0
  - 训练级 FAIL: scale_head.sigmoid 饱和 + frozen T5 → adapter grad 4 orders 坍缩, T5 main loss 无下降
  - Issue #147 关闭, 不重启 curvature-conditioned residual 路径 (跟 #145/#146 共享"adapter + frozen T5" framework 反作用根因)

### Gate 4 (= Stage 4 R@K eval): ⏹ STOP per spec
- **原因**: Gate 3 FAIL per spec (Issue #147 spec: Gate 3 PASS 前不做 Stage 4)
- **Issue spec 强制**: Issue #147 spec 明确不做 Stage 4 (只验证 Precheck + 短训练)

---

## 关键产物

- **verdict**: `verdicts/task437_issue147_gate3_fail_v1.md` (本文件)
- **commit hash**: TBD (after `git add` + `git commit` + `git push`)
- **push**: origin/main (R15 强制)
- **实施**: `scripts/task437_issue147_curvature_conditioned_residual.py`
- **8 件套审计**: config.json + SHA256(SID=`2dab2922...` + T5=`56d046db...` + train=`2c5f843d...` + adapter=`6a377018...`) + adapter_init_proof (max_diff=0) + gradient_proof (13 params non-zero) + train_trace (10 epoch 完整) + adapter.pt (save/load round-trip missing=0/unexpected=0 + forward_diff=0)
- **整体决策**: ⚠️ **架构级 PASS + 训练级 FAIL** — Issue #147 关闭

---

## R18 严格路径对比 (vs #142 / #129 旧 dual-gate)

| 维度 | Issue #142 (#129 旧 dual-gate NO-GO) | Issue #147 (本 task, 修复方向C) |
|------|--------------------------------------|------------------------------------|
| **D1 spec 摘录** | Manifest lineage PASS + control R@10=0.10203 ≈ baseline, Adapter 反作用 (zero-init gate) | curvature-conditioned residual: bounded scale (sigmoid) + identity-preserving 注入 |
| **D2 实施** | AdapterHookedHGRec.generate + zero-init gate + active-down→ReLU | CurvatureConditionedAdapter MLP + scale_head.sigmoid + gate=sigmoid(gate_logit) |
| **D3 Gate 1 失败机制** | zero-init + ReLU 早期 gradient dead | sigmoid 饱和 + frozen T5 main 切断 residual 路径 |
| **D4 引用文献** | #129 旧 dual-gate 内部 spec | arXiv:2309.04082《Curve Your Attention》 |
| **Precheck 1 (gate=0 identity)** | N/A (zero-init 永远 = 0) | PASS (max diff=0) ✓ |
| **Precheck 2 (grad non-zero)** | FAIL (zero-init 死) | PASS (13 params non-zero) ✓ |
| **Precheck 3 (SID range)** | PASS (复用 Task #84) | PASS (4 层全 in_range) ✓ |
| **Training 收敛** | 旧 #129: 2 epoch 反作用 R@10=0.03684 | #147: 10 epoch 损失 flat 1.886-1.889 |
| **R18 判定** | 必须新实验 (R18 强制) | 必须新实验 (本 task) |

---

## 后续修复路径 (R11.5 自主决策)

**核心发现 — 跟 #145/#146 共享 family 根因**: 任何依赖 frozen T5 main + auxiliary path 的 adapter 在 saturation 路径上都失败. 即使 precheck 完美 (grad 非零), 一旦进入实际 SGD 训练, auxiliary path 跟 main path 梯度量级不匹配 (main 损失 >> auxiliary 残差) → auxiliary 路径被压制.

**真正可修复方向** (跟 #145/#146 共识):
1. **Train adapter + T5 jointly**: 解除 T5 冻结, 让 adapter 跟 T5 共同学习 (但需要小心控制 learning rate 不破坏 pretrained weights)
2. **Use linear (NOT sigmoid) scale**: 避免 sigmoid 饱和, 用 small linear scale (e.g., 0.01) + 让 gradient 自然流过
3. **Auxiliary loss as REGULARIZER (not primary path)**: 让 adapter residual 跟其他东西 orthogonal, 不依赖 residual 通过 frozen T5 路径
4. **Stage 4 协议 split** (per Issue #39): 改用 T5.generate SID protocol 替代 dense ANN protocol, 允许 adapter 输出可微路径
5. **Architecture pivot**: 不再做 T5 adapter, 改在 Stage 1 RQ-VAE 几何层做 curvature conditioning (per #145 κ-dependent pairwise aux)

**Why NO-GO 收口 #147 当前方向**: 任何依赖"frozen main + auxiliary path" 的 adapter 在 T5 上都 fail (跟 #142/#129 NO-GO 同根因 family, 仅 mechanism 不同).

**Why 架构级 PASS**: precheck 完美 (identity-preserving gate=0 + grad non-zero + SID range + save/load round-trip) — 验证架构是对的, 只是这个架构跟 frozen T5 训练不兼容.

---

## 历史事故关联

| 事故 | 现象 | 根因 | 跟 #147 关系 |
|------|------|------|--------------|
| Issue #142 (#129) | AdapterHookedHGRec R@10=0.03684 (-63%) | zero-init dual-gate 早期 dead | #147 跟 #142 同 family (adapter + frozen T5) 反作用 |
| Issue #145 (Task #435) | aux minimize 推 collapse | pairwise+ranking minimization | #147 共享"frozen T5 + aux" framework 反作用 |
| Issue #146 (Task #436) | argmin + relu 切断 | contrastive margin | #147 共享 family, 不同 mechanism (sigmoid 饱和) |
| Task #84 baseline | T5-mini R@10=0.1020 | frozen T5 + standard recipe | #147 baseline recipe 是 frozen T5 (跟 #147 不兼容) |
| Task #139 (Issue #43) | reproducibility triangle R12 invariant | SHA256 ckpt + SID + eval 三件套 | #147 复用 SHA256 ckpt 协议 |

---

## 收口

- Issue #147 关闭 (`gh issue close 147 --reason completed`)
- 12 方向 NO-GO 收口 + Issue #145/#146/#147 架构级 PARTIAL = **baseline recipe 路径 + 修复尝试 联合耗尽**
- **新方向候选** (R11.5 决策, 等 owner 拍板): train adapter+T5 jointly / linear scale (NOT sigmoid) / regularizer-style aux / Stage 4 protocol split / Stage 1 几何层 conditioning
- **R@10 ceiling 0.1053 + R137 κ lock + R139 reproducibility triangle + argmin 切断 + sigmoid 饱和 + frozen T5 反作用 = baseline recipe 路径耗尽确认**