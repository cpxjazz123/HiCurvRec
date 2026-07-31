# Task #425 / Issue #133 [方向C Gate4] Stage 4 eval for #129 dual-gate ckpt

## 目标

验证 Issue #133 [方向C] 的核心假设：**#129 dual-gate ckpt 可评测性** (从 Gate 3 pass 推进到 Gate 4 R@K).
要求:
1. SHA256 验证 Task84 T5 ckpt + #102 SID + #129 adapter 三件套完整可重现性
2. Task84 T5 config (vocab=1025, d_model=128) + Stage 4 eval (beam=20, SIDRetrievalEvaluator)
3. 2 runs: control (无 adapter) + adapter (含 DualGateAdapter)
4. 6 metrics: R@5/10/20 + NDCG@5/10/20
5. PASS 条件: BOTH runs valid AND adapter R@10 > 0.1020 (HG-Rec baseline)

## 实施核心

### `scripts/task425_issue133_dual_gate_eval.py` (~250 lines)

**Reproducibility 验证**:
- `sha256_of`: SHA256 校验 T5 ckpt (products/task84/hgrec_best.pth) + SID (Instruments_t5_hrqvae_poincare.npy) + adapter (products/task422_issue129_t5_dual_gate_ckpt/adapter_trained_2ep.pt)

**模型加载**:
- `build_t5_with_adapter`: T5Config (vocab=1025, d_model=128) + T5ForConditionalGeneration 加载 Task84 ckpt (strip "model." prefix) + DualGateAdapter 加载 #129 adapter state
- `WrappedT5Adapter`: 把 DualGateAdapter 嵌入到 T5 forward (input embedding → adapter → inputs_embeds)

**评估**:
- `SIDDataset`: (history_input, target_sid_4digits)
- `compute_metrics`: SIDRetrievalEvaluator 风格 — beam=20 generate → 全 SID 序列匹配 → R@K + NDCG@K
- HG-Rec baseline 对照表 (R@5/10/20=0.0816/0.1020/0.1279, NDCG@5/10/20=0.0690/0.0755/0.0821)

### 决策阈值
- Both runs valid AND adapter R@10 > 0.1020 → ✅ PASS
- 否则 → ❌ NO-GO

## 关键产物
- `scripts/task425_issue133_dual_gate_eval.py`
- `products/task425_issue133_dual_gate_eval/verdict.json`
- `verdicts/task425_issue133_gate4_result.md`

## 联立 R 规则
- R12: 强制 ckpt 保存 (#129 已经保存, 这里只读)
- R17: commit 含 Gate 4 状态 + R@10 vs baseline
- R18: 4 维度对比 (Issue #133 spec 是 Stage 4 eval 路径, 跟 #129 Gate 3 spec 不同 — 4 维度不一致, R18 强制实施)
- R19: 立即开工不等待
- R20: 4 Gate 详细内容 ≥3-5 行/Gate
- R21: commit hash 具体
- R22: 立即开工
- R7: GPU 2 并行
- R4: py_compile 验证

## 决策
- adapter R@10 > 0.1020 → ✅ PASS (GO 收口)
- adapter R@10 ≤ 0.1020 → ❌ NO-GO (但 #129 dual-gate ckpt 合同验证 PASS, Stage 4 eval 路径打开, 给后续 issue 留通路)
- any run invalid → ❌ FAIL (Stage 4 eval 不可信)
