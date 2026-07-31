# Task #380 / Issue #87 [方向C Gate2] SID与metadata对齐产物验证

**日期**: 2026-07-31
**触发**: Issue #87 [方向C Gate2] — R16 强制 GitHub OPEN 处理
**前置**: Issue #86 Gate 1 PASS 6/6 (commit 4ad7890, real_metadata_stage1_ckpt.pt 已落盘)
**任务**: 1. R18 4 维度 vs #86 对比; 2. 复用 #86 Stage 1 ckpt + Stage 2 Sinkhorn; 3. SID + metadata per-item 对齐文件; 4. Gate 2 验证
**结果**: ⏳ 进行中

---

## 1. R18 4 维度对比 (Issue #87 vs Issue #86)

| 维度 | Issue #86 (Gate1 metadata 提取 PASS) | Issue #87 (Gate2 SID metadata 对齐) | 是否一致 |
|------|------|------|------|
| **D1 spec 摘录** | Stage 1 真实 metadata 数据流 + serialize/deserialize stable | Stage 2 Sinkhorn + 4-digit SID + per-item metadata 对齐文件 | ❌ 不同 (Issue #87 是 **Stage 2**) |
| **D2 实施核心** | FreeCurvHRQVAE 50 epoch + SIDMetadata 提取 + serialize/deserialize 5/5 stable | Stage 2 Sinkhorn-Knopp + per-item metadata 序列化到 disk + batch loader 读取 | ❌ 不同 |
| **D3 失败机制** | Stage 2 util 1.56%/0.78%/0.39% (短跑坍缩) | 必须解决坍缩才能让 4-digit SID unique ≥9500 | ❌ 不同 |
| **D4 引用文献** | arXiv:2309.04082 (Curve Your Attention) | 同 | ✅ 相同 |

→ **R18 3/4 维度不一致, 必须做 Stage 2 Sinkhorn + per-item metadata 对齐 (R18 强制)**.

## 2. Issue #87 spec 要求

| 要求 | 内容 | 来源 |
|------|------|------|
| **复用 #86 Stage 1 产物** | real_metadata_stage1_ckpt.pt + SIDMetadata schema | Issue #87 spec Gate1 |
| **Stage 2 Sinkhorn + 4-digit SID** | Sinkhorn 解码 + 4-digit dedup | Issue #87 spec Gate2 |
| **SID + metadata per-item 对齐文件** | 每个 item 一个 SID + per-layer metadata | Issue #87 spec "metadata 与 SID token 一一对齐" |
| **可被 Stage3 batch loader 读取** | 输出格式兼容 Stage3 | Issue #87 spec 框架合规 |
| **Gate 2 PASS 条件** | 4-digit SID unique ≥9500/9922, L0/L1/L2 util ≥90%, 偏差 ≤5pp, metadata/SID 对齐, 无 NaN/Inf | Issue #87 spec Gate2 |
| **禁止** | 只做 metadata stub / 绕过 SID metadata / 改 embedding init / fixed hyp_c / post-step retraction | Issue #87 spec 框架合规 |

## 3. Issue #87 Gate 决策 (R17/§19)

| R17 Gate | 决策 | 状态 |
|------|------|------|
| Gate 1 (= Stage 1) | ⏸ 已 PASS (Issue #86 闭环, 复用 ckpt) | per Issue #87 spec |
| **Gate 2 (= Stage 2 Sinkhorn)** | ⏳ 进行中 | Stage 2 Sinkhorn + per-item metadata 对齐 |
| Gate 3 (= Stage 3 T5-mini) | ⏸ STOP per spec | Issue #87 spec "Gate2 未 PASS 前禁止 Stage 3" |
| Gate 4 (= Stage 4 R@K eval) | ⏸ STOP per spec | Issue #87 spec |

---

result: Issue #87 [方向C Gate2 SID metadata 对齐] R22 + R19 立即开工 (GPU 0). 3 步: 1) 复用 #86 ckpt + Stage 2 Sinkhorn; 2) per-item metadata 对齐文件 + batch loader 格式; 3) Gate 2 验证 + commit + push + comment(含 hash) + close. ⏳ 进行中.