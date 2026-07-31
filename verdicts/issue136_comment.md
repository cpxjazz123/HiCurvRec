## Issue #136 R18+R20+R21 实证闭环 — 4 Gate 详细内容回答 (R17 + R20 + R21 强制)

### Gate 0 (= 协议重建): ✅ PASS
- 关键数据: GenRecDataset(test.parquet, code_path=SID, mode='evaluation', codebook_size=[32,64,256,1], max_len=20) test_dataset size=24772
- SHA256: T5 ckpt `56d046dbabdb1930...`, SID `2dab29229c36a969...`, Adapter `2ab2ed89d53170df...`, test.parquet `70f00f3a3ac4cf39...`, train.parquet `2c5f843d456a42ee...`
- SID shape (9922, 4), unique 4-digit = 9922/9922 (100% unique)

### Gate 1 (= Stage 1 RQ-VAE): ⏸ SKIP per spec
- 原因: 复用 task84 产物

### Gate 2 (= Stage 2 Sinkhorn): ⏸ SKIP per spec
- 原因: 复用 task84 SID npy

### Gate 3 (= Stage 3 T5-mini): ⏸ SKIP per spec
- 原因: 复用 task84 HG_Rec_best.pth

### Gate 4 (= Stage 4 R@K eval): ❌ FAIL (R@K=0)
- 关键数据: Control + Adapter 双次 Recall@5/10/20 = 0.0/0.0/0.0, NDCG=0.0
- Adapter state_dict load missing=0, unexpected=0 (100% 加载成功)
- adapter_r10 = 0.0, baseline_r10 = 0.1020 → adapter_pass = False
- 失败原因: **WrappedHGRec.forward 路由 bug** (`self.model.model(...)` 是 T5ForConditionalGeneration inner, 而 evaluate 期望 HG_Rec wrapper 路径 `self.model(input_ids=...)`). Adapter generate() 路径走 `model.model.generate(inputs_embeds=...)` 跟 HG_Rec.generate() 接口不兼容.
- 实施: scripts/task428_issue136_stage4_rebuild.py (~250 lines)

### 关键产物
- commit hash: 600c1a3
- push: origin/main
- verdict: verdicts/task428_issue136_gate4_fail_v3.md
- 整体决策: ❌ Gate 4 FAIL (WrappedHGRec routing bug, R@K=0)
- 联立 #133 → #136 = **dual-gate adapter 在 T5 generate() 路径 integration 需要 HG_Rec wrapper 配合**. R@10 ceiling 0.1053 锁死后, dual-gate 路径工程不兼容是第 6 方向 NO-GO 收口.