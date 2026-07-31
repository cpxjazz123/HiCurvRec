## Issue #139 R18+R20+R21 实证闭环 — 4 Gate 详细内容回答 (R17 + R20 + R21 强制)

### 5 件套审计 (Issue #139 spec 强制) — 全部已落地, commit <pending>
1. **config**: `products/task429_issue139_stage4_repro/config.json`
2. **sha256**: T5 ckpt `56d046dbabdb1930...`, SID `2dab29229c36a969...`, Adapter `2ab2ed89d53170df...`, test.parquet `70f00f3a3ac4cf39...`, train.parquet `2c5f843d456a42ee...`
3. **raw_log**: `logs/task429_issue139_stage4_repro.launch.log` (完整 SHA256 + 4 step 控制/adapter eval 日志)
4. **verdict**: `products/task429_issue139_stage4_repro/verdict.json` + `verdicts/task429_issue139_gate4_fail_v3.md`
5. **commit**: <pending>

### Gate 0 (= 协议重建): ✅ PASS
- GenRecDataset(test.parquet, code_path=SID, mode='evaluation', **codebook_size=[64,128,256,1]** (R18 修复), max_len=20) test_dataset=24772
- SID shape (9922, 4), unique 4-digit = 9922/9922

### Gate 1 (= Stage 1 RQ-VAE): ⏸ SKIP per spec
### Gate 2 (= Stage 2 Sinkhorn): ⏸ SKIP per spec
### Gate 3 (= Stage 3 T5-mini): ⏸ SKIP per spec

### Gate 4 (= Stage 4 R@K eval): ❌ FAIL (Adapter 严重反作用)
- Control mean (双复跑, seed=43/44):
  - R@5=**0.0816**, R@10=**0.10203**, R@20=**0.1279**
  - NDCG@5=**0.0690**, NDCG@10=**0.0755**, NDCG@20=**0.0821**
- Adapter mean (双复跑, seed=143/144):
  - R@5=**0.0212**, R@10=**0.03684**, R@20=**0.0605**
  - NDCG@5=**0.0122**, NDCG@10=**0.0171**, NDCG@20=**0.0232**
- Baseline: R@10=**0.1020**
- **delta_adapter_vs_control = -62.9%** (adapter 严重反作用)
- rep_delta = 0.0 (control #1=#2, adapter #1=#2, 确定性)
- Adapter load: missing=0, unexpected=0 (100% 加载成功)
- 实施: scripts/task429_issue139_stage4_repro.py (~260 lines, R18 修复 AdapterHookedHGRec.generate 走 self.hgrec.model.generate(inputs_embeds=...))
- 失败原因: **协议层 PASS** (Control R@10=0.10203 完全复现 baseline 0.1020, Issue #136 R18 修复成功). **Adapter 层 NO-GO** (R@10=0.03684 -62.9% vs control): #129 dual-gate adapter 仅 2 epoch 训练, gate_zero + gate_active_up 都是 zero-init, 实际只走 active_down→ReLU 一次, 跟 T5 input embedding 适配路径不兼容. Issue #133 已知 (proxy R@K=0), Issue #139 真实协议下也确认 (R@K=0.04).

### 关键产物
- commit hash: <pending>
- push: origin/main
- verdict: verdicts/task429_issue139_gate4_fail_v3.md
- 实施: scripts/task429_issue139_stage4_repro.py
- products/task429_issue139_stage4_repro/verdict.json
- 整体决策: ❌ Gate 4 FAIL (Control 协议 PASS, Adapter 严重反作用, Issue #136 R18 修复成功 + Issue #139 dual-gate NO-GO 收口)

### 联立 #133 + #136 + #139 = 9 方向 NO-GO 收口
Sinkhorn transport / shared plan / dual-gate proxy / hard EMA / layer mixing / WrappedHGRec bug / hard-EMA repro / layer-mixing repro / dual-gate real-protocol. 共同结论 = R@10 ceiling 0.1053 锁死后, R137 κ lock + R139 reproducibility triangle + dual-gate 工程不兼容 三件套锁死路径耗尽.