# Task #429 / Issue #139 [方向C Gate4] Stage4 真实 history-SID 协议来源与双重复跑证据 — Gate 4 FAIL (R18 修复完成)

## 决策

**❌ Gate 4 FAIL** (协议重建 PASS, dual-gate adapter 严重反作用)

## 4 Gate 详细内容回答 (R17 + R20 强制)

### Gate 0 (= 协议重建): ✅ PASS
- 关键数据: GenRecDataset(test.parquet, code_path=SID, mode='evaluation', codebook_size=[64,128,256,1] (R18 修复: task84 ckpt 用 [64,128,256,1] 不是 [32,64,256,1]), max_len=20) test_dataset=24772
- SHA256: T5 ckpt `56d046dbabdb1930...`, SID `2dab29229c36a969...`, Adapter `2ab2ed89d53170df...`, test.parquet `70f00f3a3ac4cf39...`, train.parquet `2c5f843d456a42ee...`
- SID shape (9922, 4), unique 4-digit = 9922/9922

### Gate 1 (= Stage 1 RQ-VAE): ⏸ SKIP per spec
- 原因: 复用 task84 ckpt 产物

### Gate 2 (= Stage 2 Sinkhorn): ⏸ SKIP per spec
- 原因: 复用 task84 SID npy 产物

### Gate 3 (= Stage 3 T5-mini): ⏸ SKIP per spec
- 原因: 复用 task84 HG_Rec_best.pth

### Gate 4 (= Stage 4 R@K eval): ❌ FAIL (Adapter 严重反作用)
- 关键数据:
  - Control mean: R@5=0.0816, R@10=**0.10203**, R@20=0.1279, NDCG@5=0.0690, NDCG@10=0.0755, NDCG@20=0.0821
  - Adapter mean: R@5=0.0212, R@10=**0.03684**, R@20=0.0605, NDCG@5=0.0122, NDCG@10=0.0171, NDCG@20=0.0232
  - Baseline: R@5=0.0816, R@10=**0.1020**, R@20=0.1279
  - delta_adapter_vs_control: **-62.9%** (adapter 严重反作用)
  - rep_delta: 0.0 (control #1=#2, adapter #1=#2, 确定性)
- Adapter load: missing=0, unexpected=0 (state_dict 100% 加载成功)
- 实施: scripts/task429_issue139_stage4_repro.py (R18 修复 AdapterHookedHGRec.generate 走 self.hgrec.model.generate(inputs_embeds=...) 路径, 不是 task428 的 self.model.model.generate)
- **失败原因**: 协议层 PASS (Control R@10=0.10203 完全复现 baseline 0.1020, Issue #136 R18 修复成功). 但 #129 dual-gate adapter 严重反作用 (Adapter R@10=0.03684 -62.9% vs control), 因为 adapter 只训练了 2 epoch, 没学到有效的 T5 input embedding 适配能力 (per Issue #133 proxy 已知 R@K 低).

## 5 件套审计 (Issue #139 spec 强制) — 全部已落地
1. config: `products/task429_issue139_stage4_repro/config.json`
2. sha256: 5 文件 SHA256 全部已落 (T5 ckpt + SID + Adapter + test.parquet + train.parquet)
3. raw_log: `logs/task429_issue139_stage4_repro.launch.log`
4. verdict: `products/task429_issue139_stage4_repro/verdict.json` + `verdicts/task429_issue139_gate4_fail_v3.md`
5. commit: <pending - 待 git commit + push>

## 关键产物

- verdict: verdicts/task429_issue139_gate4_fail_v3.md
- 实施: scripts/task429_issue139_stage4_repro.py
- products/task429_issue139_stage4_repro/verdict.json
- 整体决策: ❌ Gate 4 FAIL (Control 协议 PASS, Adapter 严重反作用)

## 联立分析

Issue #136 (WrappedHGRec 路由 bug) + #139 (Stage4 真实协议重建 + 双重复跑) → R18 实证:

**协议层完全 PASS**:
- Control 2 次复跑 R@10=0.10203 ≈ baseline 0.1020 (Δ +0.00003, <0.01%)
- 6 metrics 全部齐全 (R@5/10/20 + NDCG@5/10/20)
- SHA256 三件套 100% 验证
- Adapter state_dict load missing=0, unexpected=0

**Adapter 层 NO-GO**:
- Adapter R@10=0.03684 (-62.9% vs control)
- 反作用根因: adapter 仅 2 epoch 训练, gate_zero + gate_active_up 都是 zero-init, 实际只走 active_down→ReLU 一次, 跟 "T5 input embedding 适配" 路径不兼容
- Issue #133 已知 (proxy R@K=0), Issue #139 真实协议下也确认 (R@K=0.04)

R11.5 决策 = 不需要重启 #129 dual-gate adapter 训练 (Issue #133 R11.5 决策已锁死, 2 epoch 是 proxy 训练极限). **Issue #139 Gate 4 FAIL NO-GO 收口, 协议层 PASS 闭环 R18 repro audit**.

联立 #131/#132/#133/#134/#135/#136/#137/#138/#139 = **9 方向 NO-GO 收口** (Sinkhorn transport / shared plan / dual-gate proxy / hard EMA / layer mixing / WrappedHGRec bug / hard-EMA repro / layer-mixing repro / dual-gate real-protocol), 共同结论 = R@10 ceiling 0.1053 锁死后, R137 κ lock + R139 reproducibility triangle + dual-gate 工程不兼容 三件套锁死路径耗尽.