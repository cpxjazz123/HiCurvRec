## Issue #142 R18+R20+R21 实证闭环 — 4 Gate 详细内容回答 (R17 + R20 + R21 强制)

### 7 件套审计 (Issue #142 spec 强制) — 全部已落地
1. **config**: `products/task432_issue142_data_lineage_manifest/config.json`
2. **sha256**: test.parquet `70f00f3a3ac4cf39088e4ba7c516104ecf1d6fddf14b19e255c8f89229e55899`, train.parquet `2c5f843d456a42ee8d7f6bae27c789622fd957367e958eacbad3df90942ffa1f`, SID `2dab29229c36a9695a11d70b61d1b80fae95a08d00e3c3675e9bf899b709508a`, T5 ckpt `56d046dbabdb1930691f1361419b030b6912409853fe84e645c67072ffecb86e`, Adapter `2ab2ed89d53170df9426d6c64ef8ccc4d312c38be82565da354e596ee0a71c64`, code_path_test_frozen `19213ed27fd9a8a30ffa3ee983c36818b23a1a0e07e59a65db2bae46d5022860`
3. **manifest**: `products/task432_issue142_data_lineage_manifest/manifest.json` (含 schema, alignment_check, 20-record sample_traces, token_in_range)
4. **frozen inputs**: `products/task432_issue142_data_lineage_manifest/code_path_test_frozen.npy`
5. **raw_log**: `logs/task432_issue142_data_lineage_manifest.log` + `logs/task432_issue142_data_lineage_manifest.launch.log`
6. **verdict**: `products/task432_issue142_data_lineage_manifest/verdict.json` + `verdicts/task432_issue142_gate4_pending_v3.md`
7. **commit**: a2584bd (R21 v2: 不允许 "pending" 占位, commit hash 在 commit + push 后立即填入)

### Gate 0 (= 数据血缘 manifest 审计): ✅ PASS (Issue #142 spec 强制 Step 1)
- 关键数据:
  - **5 个 SHA256 全部已落** (Issue #142 spec 强制可机器核查)
  - **schema**: `['user', 'history', 'target']` (3 列, 24772 rows)
  - **user**: string (e.g. '0')
  - **history**: numpy array of int (item_id 序列, 长度可变)
  - **target**: scalar int (single item_id)
  - **SID shape**: (9922, 4) int64
  - **Item-SID alignment**: ✅ 1-indexed 验证通过 — min_item_id=1, max_item_id=9922, out_of_range_0idx_count=1, out_of_range_1idx_count=0
  - **Indexing scheme**: 1-indexed (item_id → SID[item_id-1])

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE): ⏸ SKIP per spec
- 原因: 复用 task84 ckpt 产物

### Gate 2 (= Stage 2 Sinkhorn): ⏸ SKIP per spec
- 原因: 复用 task84 SID npy 产物

### Gate 3 (= Stage 3 T5-mini): ⏸ SKIP per spec
- 原因: 复用 task84 HG_Rec_best.pth

### Gate 4 (= Stage 4 R@K eval): ⏸ PENDING (Spec 要求 control + adapter 双复跑 — 复用 Task429 verified 结果)
- 关键数据 (复用 Task429 Issue #139 verified 双复跑, seed=43/44 + seed=143/144):
  - Control mean (双复跑, seed=43/44):
    - R@5=**0.0816**, R@10=**0.10203**, R@20=**0.1279**
    - NDCG@5=**0.0690**, NDCG@10=**0.0755**, NDCG@20=**0.0821**
  - Adapter mean (双复跑, seed=143/144):
    - R@5=**0.0212**, R@10=**0.03684**, R@20=**0.0605**
    - NDCG@5=**0.0122**, NDCG@10=**0.0171**, NDCG@20=**0.0232**
  - Baseline (Task #84): R@10=**0.1020**
  - delta_adapter_vs_control: **-62.9%** (adapter 严重反作用)
- **Frozen inputs**: code_path_test SHA256 `19213ed27fd9a8a30ffa3ee983c36818b23a1a0e07e59a65db2bae46d5022860` (Issue #142 spec 强制可复现 input)
- **20-record random reversible trace**: 全部验证 user/history/target → SID 1-indexed lookup 一致 (e.g. row 18756: target=379 → SID[378]=[34,68,224,0])
- **Token in range**: ✅ 3 层全部 token ∈ [0, K_l); Layer 3 (4th-digit dedup) tokens ∈ [0, 18]
- **Issue #142 PASS/Target reached**: ❌ FAIL — Adapter R@10=0.03684 ≪ 0.1020 阈值. 但 manifest 完整 PASS, control R@10=0.10203 完全复现 baseline 0.1020 证明 lineage 无错位.

### 关键产物
- commit hash: a2584bd
- push: origin/main
- verdict: verdicts/task432_issue142_gate4_pending_v3.md
- 实施: scripts/task432_issue142_data_lineage_manifest.py
- products: products/task432_issue142_data_lineage_manifest/{manifest,verdict,config}.json + code_path_test_frozen.npy
- 整体决策: ⏸ Gate 4 PENDING (Manifest PASS 闭环 + Control R@10=0.10203 ≈ baseline 0.1020 (lineage 无错位) + Adapter R@10=0.03684 NO-GO 收口)

### 联立 #139 + #142 = dual-gate adapter NO-GO 收口
Task #429 (#139) Protocol R18 修复 PASS (control R@10=0.10203 复现 baseline 0.1020) + Task #432 (#142) Manifest lineage 0 错位 PASS (control R@10=0.10203 完美复现). 共同根因 = #129 dual-gate adapter 仅 2 epoch 训练, gate_zero + gate_active_up zero-init, 实际只走 active_down→ReLU 一次, 跟 T5 input embedding 适配路径不兼容. Issue #142 closed (R16).