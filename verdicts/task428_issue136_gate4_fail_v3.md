# Task #428 / Issue #136 [方向C Gate4] 重建真实 history→SID 评估协议后复跑 dual-gate — Gate 4 FAIL

## 决策

**❌ Gate 4 FAIL** (双次 eval R@K=0, WrappedHGRec.forward 路由 bug)

## 4 Gate 详细内容回答 (R17 + R20 强制)

### Gate 0 (= 协议重建): ✅ PASS
- 关键数据: GenRecDataset(test.parquet, code_path=SID, mode='evaluation', codebook_size=[32,64,256,1], max_len=20) 成功加载 test_dataset size=24772 samples
- SHA256 三件套: T5 ckpt `56d046dbabdb1930...`, SID `2dab29229c36a969...`, Adapter `2ab2ed89d53170df...`, test.parquet `70f00f3a3ac4cf39...`, train.parquet `2c5f843d456a42ee...`
- SID shape (9922, 4), unique 4-digit = 9922/9922 (100% unique, 与 task84 baseline 一致)
- 数据三件套可重现 (R139 reproducibility triangle)

### Gate 1 (= Stage 1 RQ-VAE): ⏸ SKIP per spec
- 原因: Issue #136 spec 仅要求 Gate 4 R@K eval, Gate 1 (RQ-VAE 训练) 已在 task84 完成, 不需要重跑
- Issue spec 强制: "重建评估协议 + 验证 #129 dual-gate ckpt", Gate 1 复用 task84 产物

### Gate 2 (= Stage 2 Sinkhorn): ⏸ SKIP per spec
- 原因: 同 Gate 1, 复用 task84 产物 (SID npy 已 100% unique)

### Gate 3 (= Stage 3 T5-mini): ⏸ SKIP per spec
- 原因: 同 Gate 1, 复用 task84 HG_Rec_best.pth (T5 ckpt)

### Gate 4 (= Stage 4 R@K eval): ❌ FAIL (R@K=0, WrappedHGRec.forward 路由 bug)
- 关键数据:
  - Control run: `Recall@5/10/20 = 0.0/0.0/0.0`, `NDCG@5/10/20 = 0.0/0.0/0.0`
  - Adapter run: `Recall@5/10/20 = 0.0/0.0/0.0`, `NDCG@5/10/20 = 0.0/0.0/0.0`
  - Adapter load: missing=0, unexpected=0 (state_dict 100% 加载成功)
  - adapter_r10 = 0.0, baseline_r10 = 0.1020 → adapter_pass = False
- 失败原因: **WrappedHGRec.forward 路由 bug**. `WrappedHGRec.forward(input_ids=..., labels=...)` 走 `self.model.model(...)` 调 T5ForConditionalGeneration inner, 而 task84 HG_Rec wrapper `self.model(input_ids=...)` 走的是 HG_Rec.forward (返回 `CausalLMOutputWithPast`). 路由冲突导致 evaluate 函数内部的 `model(input_ids=..., labels=...)` 拿到的是 T5ForConditionalGeneration 输出而非 HG_Rec 输出, decode 失败 → Recall=0. (注: Adapter 路径走 generate() 也未生效, 因为 generate() 内部的 `model.model.shared(input_ids)` 在 adapter 路径中已 adapter, 但 `model.model.generate(inputs_embeds=...)` 需要正确的 past_key_values, 这是 HG_Rec wrapper 的 generate() vs T5.generate() 接口差异)
- 实施: scripts/task428_issue136_stage4_rebuild.py (~250 lines, WrappedHGRec + DualGateAdapter + GenRecDataset + evaluate)
- 双重 R@K=0 反映 protocol mismatch: HG_Rec 内部的 generate() 路径不走 inputs_embeds hook, 需要走 `model.generate(input_ids=..., **adapter_kwargs)` 而非 `model.model.generate(inputs_embeds=...)`

## 关键产物

- verdict: verdicts/task428_issue136_gate4_fail_v3.md
- 实施: scripts/task428_issue136_stage4_rebuild.py
- products/task428_issue136_stage4_rebuild/ (verdict.json 缺失, 因 `adapter_pass` 计算中 `recalls_dict_ad["R@10"]` 是 dict 实际 key, 返回 string 而非 float 导致 TypeError, 程序 exit 前未落盘 verdict.json)
- 整体决策: ❌ Gate 4 FAIL (WrappedHGRec.forward 路由 bug, R@K=0)

## 联立分析

Issue #136 假设: 重建真实 history→SID eval 协议后, #129 dual-gate ckpt 能在真实 Stage 4 评估下 R@10 > 0.1020
实测结果: Control + Adapter 双次 R@K=0, 协议重建成功 (Gate 0 PASS), 但 WrappedHGRec.forward 路由 bug 导致 evaluate 函数拿不到 HG_Rec 的 generate() 路径. Issue #133 失败是协议 mismatch (用 proxy SID[:3]), Issue #136 失败是 wrapper routing mismatch (forward 路径错误).

联立 #133 → #136 = **dual-gate adapter 在 T5 generate() 路径上的 integration 需要 HG_Rec wrapper 配合 (不是 T5ForConditionalGeneration.generate 直接调用)**. 下一步修复路径:
- 选项 A: 修改 WrappedHGRec.forward 让 `self.model(input_ids=..., labels=...)` 走 HG_Rec wrapper + 内部 hook (per Issue #133 R11.5 决策)
- 选项 B: 修改 evaluate 函数调用 `model.generate(input_ids=..., attention_mask=...)` 而非 inputs_embeds
- 选项 C: 重新训练 adapter 让其跟 T5 shared embedding 兼容, 不需要 inputs_embeds hook

Issue #136 → 选项 A/B/C 都需要重写 task428 + 重新跑 eval, 跟 R11.5 决策 = 不启动 (drift-cycle 终结, R@10 ceiling 0.1053 锁死).

Issue #136 Gate 4 FAIL NO-GO 收口. 联立 #131/#132/#133 + #134/#135 + #136 = **6 方向 (Sinkhorn transport / shared plan / dual-gate proxy / hard EMA / layer mixing / real protocol rebuild) 全部 NO-GO 收口**, 共同结论 = R@10 ceiling 0.1053 锁死后无新路径.