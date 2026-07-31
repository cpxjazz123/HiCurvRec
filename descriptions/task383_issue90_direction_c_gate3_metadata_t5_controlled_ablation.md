# Task #383 / Issue #90 [方向C Gate3] metadata 条件信号进入 T5 受控消融 C0/C1/C2

**日期**: 2026-07-31
**触发**: Issue #90 [方向C Gate3] metadata 条件信号进入 T5 受控消融 — R16 + R22 强制立即开工
**前置**: Issue #86 Gate 1 metadata PASS (commit 4ad7890) + Issue #87 Gate 2 SID+metadata 对齐 FAIL NO-GO (commit 5332c0b, SID unique=256/9922)
**任务**: 1. R18 4 维度对比 vs Issue #87; 2. Stage 3 T5-mini C0/C1/C2 受控消融; 3. 报告 梯度/on-off 差异/打乱破坏; 4. Gate 3 验证
**结果**: ⏳ 进行中

---

## 1. R18 4 维度对比 (Issue #90 vs Issue #87)

| 维度 | Issue #87 (Stage 2 SID+metadata 对齐 FAIL) | Issue #90 (Stage 3 T5 metadata 受控消融) | 是否一致 |
|------|------|------|------|
| **D1 spec 摘录** | Stage 2 Sinkhorn + per-item metadata 对齐文件 | Stage 3 T5-mini C0/C1/C2 受控消融 (metadata off / embedding / attention-bias) | ❌ 不同 Gate (Stage 2 vs Stage 3) |
| **D2 实施核心** | FreeCurvHRQVAE Stage 1 ckpt + Sinkhorn + sid_metadata.json | T5-mini 训练 + metadata embedding/attention-bias 模块 + 打乱破坏验证 | ❌ 不同 |
| **D3 失败机制** | SID 坍缩导致 metadata 对齐失败 | T5 能否从 learned metadata 提取非零可证伪几何信号 | ❌ 不同 |
| **D4 引用文献** | arXiv:2309.04082 | 同 | ✅ 相同 |

→ **R18 3/4 维度不一致, 必须做 Stage 3 受控消融实证 (R18 强制)**.

## 2. Issue #90 spec 要求

| 要求 | 内容 | 来源 |
|------|------|------|
| **复用 #86/#87 产物** | real_metadata_stage1_ckpt.pt + sid_metadata.json | Issue #90 Gate1 spec |
| **SID unique < 9500 显式标注** | verdict 必须显式写"受控消融, 非完整 pipeline PASS" | Issue #90 Gate2 spec |
| **C0** | metadata off, 等价 vanilla T5 输入 | Issue #90 Gate3 spec |
| **C1** | metadata embedding only, 读取 kappa_l/scale_l/confidence/mask | Issue #90 Gate3 spec |
| **C2** | attention-bias on, metadata 影响 attention logits 或 representation mixing | Issue #90 Gate3 spec |
| **报告** | 梯度范数, 开关 on/off logits 差异, 打乱 metadata 后指标变化, padding mask, training/val loss | Issue #90 Gate3 spec |
| **PASS** | C1/C2 相比 C0 有非零可重复响应; 打乱 metadata 破坏响应; 梯度有限非零; 无 NaN/Inf | Issue #90 Gate3 spec |
| **Gate 4 STOP** | Gate 3 PASS 后才允许, 必须统一报告 C0/C1/C2 6 项指标 | Issue #90 Gate4 spec |

## 3. Issue #90 Gate 决策 (R17/§19)

| R17 Gate | 决策 | 状态 |
|------|------|------|
| Gate 1 (= Stage 1) | ✅ 复用 #86 PASS (commit 4ad7890) | per Issue #90 spec |
| Gate 2 (= Stage 2) | ❌ FAIL (复用 #87, SID unique=256/9922) | per Issue #90 spec "显式标注受控消融" |
| **Gate 3 (= Stage 3 T5)** | ⏳ 进行中 | C0/C1/C2 3 变体训练 |
| Gate 4 (= Stage 4 R@K eval) | ⏸ STOP per Issue #90 spec | Gate 3 PASS 后才允许 |

---

## 4. 实施策略 (R11.5 自主决策)

- 复用 task83 P5-SID 训练模式 + task84 stage3 T5-mini 训练脚本
- C0/C1/C2 三个受控变体, 每个 50 epoch T5-mini 训练 (~ 1.5h on L40S)
- 并行启动 3 个变体 (GPU 0/2/3, GPU 1 备用), R7 4 GPU 空闲
- 报告: 梯度范数 + on/off logits 差异 + 打乱 metadata 指标变化 + 训练/val loss
- ckpt 落盘 (R12 强制)

---

result: Issue #90 [方向C Gate3 metadata 条件信号进入 T5 受控消融 C0/C1/C2] R22 + R19 立即开工 (GPU 0/2/3 并行). 3 步: 1) 复用 #86 ckpt + T5-mini 3 变体训练; 2) 报告梯度/on-off 差异/打乱破坏; 3) Gate 3 验证 + commit + push + comment(含 hash) + close. ⏳ 进行中.