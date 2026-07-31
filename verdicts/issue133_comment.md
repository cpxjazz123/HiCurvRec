## Issue #133 R18+R20+R21 实证闭环 — 4 Gate 详细内容回答 (R17 + R20 + R21 强制)

### Gate 1 (= Stage 1 RQ-VAE): ✅ PASS per spec (复核 task84 ckpt)
- 关键数据: Task84 ckpt SHA256 `56d046dbabdb1930691f1361419b030b6912409853fe84e645c67072ffecb86e` ✅ match

### Gate 2 (= Stage 2 Sinkhorn): ✅ PASS per spec (复核 task84 SID)
- 关键数据: Task84 SID SHA256 `2dab29229c36a9695a11d70b61d1b80fae95a08d00e3c3675e9bf899b709508a` ✅ match (shape (9922, 4) int64, unique 9922/9922)

### Gate 3 (= Stage 3 T5-mini dual-gate): ✅ PASS (复用 #129 结论)
- 关键数据: T5 ckpt load: missing=0, unexpected=0 (T5Config num_layers=6/num_decoder_layers=4/d_model=128/d_ff=1024/num_heads=6/d_kv=64/vocab=1025 match task84 ckpt exactly); DualGateAdapter 32768 params (gate_zero init zeros, gate_active_down/up); Adapter load: missing=0, unexpected=0 (task422 adapter_trained_2ep.pt 含 adapter_state_dict 子键)

### Gate 4 (= Stage 4 R@K eval): ⚠️ PARTIAL FAIL (SHA256 + load 全 PASS, 协议不匹配导致 R@10=0)
- 关键数据: Both runs valid=True (control + adapter 都成功 generate, n_valid=9922/9922); **Adapter R@10 = 0.0000** vs baseline 0.1020 (FAIL); Control R@10 = 0.0000 vs baseline 0.1020 (FAIL)
- **失败原因**: **评估协议不匹配** — HG_Rec_best.pth 是 Stage 4 trained SID generator, 训练用真实 user history sequence → SID. 项目内没有可用的 user history dataset, 当前 SIDDataset 用 SID[:3] (3 digits) → 推断 next SID 是结构性 proxy, 不代表真实 Stage 4 eval 协议
- 实施: scripts/task425_issue133_dual_gate_eval.py (~280 lines)
- verdict 路径: verdicts/task425_issue133_gate4_fail_v3.md

### 关键产物
- commit hash: 1f63b00
- push: origin/main (pushed 2026-08-01)
- verdict: verdicts/task425_issue133_gate4_fail_v3.md
- 实施: scripts/task425_issue133_dual_gate_eval.py
- 整体决策: ⚠️ Gate 4 PARTIAL — **byte-perfect 验证完整 PASS, Stage 4 真实评估协议需要 user history dataset 重建 (不在当前 products/), 当前 proxy 协议不可判定 adapter 优劣**
- 联立 #129 → #133: #129 dual-gate ckpt byte-perfect PASS, #133 SHA256 + load 完整 PASS, 但 Stage 4 R@K 评估需要重建 dataset. 后续 issue 必须先重建 Musical_Instruments Stage 4 dataset (history→SID) 才能继续 R@K eval
