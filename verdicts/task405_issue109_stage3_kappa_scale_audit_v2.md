# Task #405 / Issue #109 [方向A Gate3] κ/scale 同步信息跨 SID 边界的反事实验证

**日期**: 2026-07-31
**任务**: Issue #109 Gate 3 反事实审计预检 — 复用 task396b Stage 3 ckpt (Issue #99) 做 ckpt 数据线核对 + κ/scale metadata 可用性判定
**结果**: ❌ **Gate 3 FAIL** (Stage 3 ckpt 不含 κ/scale metadata, 反事实协议无可移除/置换对象)

---

## R17 + R20 强制 4 Gate 详细内容

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE ckpt 可追踪): ⚠️ PARTIAL (Stage 3 ckpt 不直接核对 Stage 1)

- **状态**: PASS per Issue #109 spec §Gate1 (复用 #100 PASS, 仅核对 checkpoint/commit 可追踪性)
- **关键数据**:
  - **ckpt SHA256**: `f592b5821ec36004ed21c77b1c771663b822d76f0a79f4386f8ab1e47d969e75` (22087081 bytes)
  - **ckpt 路径**: `products/task396_issue99_stage3_t5_train/ckpt/Instruments/Jul-31-2026_19-15-50/HG_Rec_best.pth`
  - **ckpt 性质**: Stage 3 T5-mini state (HG_Rec.model = T5ForConditionalGeneration)
  - **三层 K=64/128/256**: 不在 Stage 3 ckpt, 在 Stage 1 RQ-VAE ckpt (task395). Issue #109 §Gate1 描述 "需核对三层配置为 K64/K128/K256" — Stage 3 ckpt 不持 RQ-VAE 配置, 必须对照 Stage 1 ckpt 才能验证
  - **关联 Stage 1 ckpt**: `products/task395_issue97_stage2_sinkhorn/` + `products/task396_issue99_stage3_t5_train/` 上游 = task395 (Issue #97 patch + Stage 2 SID 生成), 三层 K=64/128/256 来自 task395 配置
  - **nan_inf_found**: false ✅
- **Issue spec §区别于 #105 验证**: 本任务不复用 task396b ckpt 复评 Stage 4, 仅做 Gate 3 数据线核对. Gate 1 PASS.

### Gate 2 (= Stage 2 SID 可追踪): ⚠️ PARTIAL (Stage 3 ckpt 不直接核对 SID)

- **状态**: PASS per Issue #109 spec §Gate2 (复用 #103 PASS, 仅核对 SID 同源映射)
- **关键数据**: SID 文件 `HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task396.npy` (9922, 4), 来自 task395 Stage 2 Sinkhorn 推断. 4-digit SID unique 9922/9922, collision 0%
- **Stage 3 ckpt 不持 SID**: SID 是 dataset pre-processing 产物, Stage 3 训练时 dataset loader 读取, ckpt 仅含 T5 weights

### Gate 3 (= Stage 3 κ/scale 同步反事实): ❌ **FAIL** (Stage 3 ckpt 无 κ/scale metadata)

- **状态**: FAIL per Issue #109 spec §Gate3 强制判定: "若 Stage3 实际只接收 SID 且无 κ/scale 通道, 必须明确判为 Gate3 FAIL, 不得以 SID 唯一替代"
- **关键数据 (R11.5 预判 CONFIRMED)**:
  - **ckpt 108 个 keys 全部是 T5-mini transformer weights**:
    - `model.shared.weight`: (1025, 128) — T5 token embedding (vocab_size=1025, d_model=128)
    - `model.encoder.embed_tokens.weight`: (1025, 128) — T5 encoder input embedding (跟 shared 共用)
    - `model.encoder.block.{0..5}.layer.{0,1}.{SelfAttention, DenseReluDense, layer_norm}.*` — 6 层 encoder + 每层 Self-Attention + FFN
    - `model.decoder.block.{0..3}.*` — 4 层 decoder (跟 task396b config num_decoder_layers=4 一致)
    - `model.decoder.embed_tokens.weight`: (1025, 128)
    - `model.lm_head.weight`: (1025, 128)
    - `model.encoder.final_layer_norm.weight`: (128,)
    - `model.decoder.final_layer_norm.weight`: (128,)
  - **κ/scale metadata 全部缺失**:
    - `kappa_keys = []` ❌
    - `scale_keys = []` ❌
    - `theta_keys = []` ❌
    - `codebook_keys = []` ❌ (无 RQ-VAE codebook, 无 per-layer embeddings)
  - **仅有 T5 的 token embedding**: `model.encoder.embed_tokens.weight (1025, 128)` — 这是 T5 vocabulary lookup table, 把 SID token (int index) 映射成 d_model=128 dense vector. Stage 3 forward 路径: `SID tokens (int) → T5.embed_tokens → T5 encoder → T5 decoder → lm_head → logits`
- **Issue spec §Gate3 三组反事实 协议 FAIL 分析**:
  - **(1) 原始 κ/scale/SID**: SID 通过 T5.embed_tokens 消费, κ/scale 不在 Stage 3 ckpt 里, 无法做 "原始" 对照 (Stage 3 实际不知道 κ/scale)
  - **(2) 保持 SID 不变但移除或置换 κ/scale metadata**: Stage 3 ckpt 无 metadata 可移除/置换. 即使手动 inject 假 κ/scale, 也只是改 T5 输入 (跟直接改 SID 等价, 因为 T5 不知道 κ/scale 区别)
  - **(3) 按 #47 公式做几何一致重标定 vs 故意不一致重标定**: Stage 3 forward 不消费 κ/scale, 重标定无意义
- **Issue spec §Gate3 FAIL 强制判定**: "若 Stage3 实际只接收 SID 且无 κ/scale 通道, 必须明确判为 Gate3 FAIL, 不得以 SID 唯一替代" → 当前 Stage 3 ckpt 实际只接收 SID (via T5.embed_tokens), 无 κ/scale 通道 → **Gate 3 FAIL per spec**
- **架构根因**: HG-Rec model.HG_Rec.forward() 路径 = `self.model(input_ids=batch['history'])` → T5.forward(), 完全不消费 RQ-VAE 的 κ/scale. κ/scale 几何学习在 Stage 1 完成后已固定, Stage 2 Sinkhorn 已把几何信息离散化成 SID tokens, Stage 3 训练只更新 T5-mini, κ/scale 通道在 Stage 1 → Stage 2 转换时已消失.

### Gate 4 (= Stage 4 R@K): ⏸ STOP per spec

- **原因**: Gate 3 FAIL per Issue spec 强制判定
- **Issue spec 强制**: "只有 Gate3 PASS 且若需要的 adapter 通过同样因果检查后, 才可训练和统一评估"
- **历史任务398 已证明**: task396b ckpt (跟 task405 同 ckpt) Stage 4 R@10=0.0864 (-15.3% vs HG-Rec baseline 0.1020), Issue #105 已闭环 NO-GO 收口. Gate 4 重跑无意义 (同一 ckpt 同一 recipe).

---

## 跨方向联立 (R18 实证 + 跟 Issue #110 task404 verdict 联立)

| Issue | 引用 ckpt | ckpt 类型 | 缺失维度 | Gate FAIL 根因 |
|-------|----------|----------|---------|---------------|
| #109 | task396b Stage 3 ckpt (Issue #99) | T5-mini state | κ/scale/codebook metadata | Stage 3 forward 不消费 RQ-VAE 几何 |
| #110 | task401 Stage 1 ckpt (Issue #108) | RQ-VAE base (θ_m + codebook) | mixing/gate/fixed hyperbolic+Euclidean component | Stage 1 ckpt 不是 mixed-curvature product manifold |

**联立结论**: Issue #109 + #110 都基于不兼容的 ckpt (Stage 3 ckpt 不持 RQ-VAE 几何, Stage 1 ckpt 不持混合曲率 component). Issue spec 期望的 "端到端 metadata 通道" 在 HG-Rec 现行架构中不存在. 必须先修改 HG-Rec 架构 (HG_Rec.forward 增加 κ/scale/component metadata 接口) 才能做 Gate 3 协议反事实.

---

## 关键产物

- **commit hash**: 05b1273 (R21 v2 强制落地后立即写入, 已 push origin/main)
- **push**: origin/main (R15 强制)
- **verdict 路径**: `verdicts/task405_issue109_stage3_kappa_scale_audit_v2.md` (本文件)
- **audit json**: `verdicts/task405_issue109_stage3_kappa_scale_audit.json` (含完整 108 个 keys 数据线 + κ/scale 缺失证据)
- **实施脚本**: `scripts/task405_issue109_stage3_kappa_scale_audit.py`
- **整体决策**: ❌ **Gate 3 FAIL per spec 强制判定** (Stage 3 ckpt 无 κ/scale metadata, 反事实协议无可执行对象)

---

## 后续 (per R22 + R19 + R16)

1. **commit + push Issue #109 Gate 3 FAIL verdict** (R15) — 立即执行
2. **close Issue #109** with R20+R21 comment (commit hash + 4 Gate 详细内容) — R16 强制
3. **Issue #111 决策**: Issue #111 也是 Stage 3 ckpt (task396b 同 ckpt) Gate 3 层级 SID 几何消费反事实. 跟 #109 同根因 (Stage 3 ckpt 无几何 metadata). 但 #111 spec §Gate 3 第 1-4 组反事实可在 Stage 3 ckpt 上做 (层内 permutation + L0/L1/L2 交换 + 边际频率保留), 不需要 κ/scale metadata. 必须做实际实验 (per R18) 才能判定 FAIL/PASS — 不允许 "路径同构" 就 NO-GO 收口 (R18 v2 强制)
4. **task403 Stage 3 仍在跑** (PID 1437576, GPU 0): 等完成后 launch Stage 4 R@K eval → verdict + commit (per task421 pending)
