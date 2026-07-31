# Task #367 / Issue #74 [方向C 预检] sid_metadata 到 attention-bias 合同

**日期**: 2026-07-31
**触发**: Issue #74 [方向C 预检] learned-κ SID metadata 进入 T5 attention-bias 接口合同 — R16 强制 GitHub OPEN 处理
**前置**: Issue #71 闭环完成 (task363 Gate 1 precheck blocked, 0/7 markers 缺失)
**任务**: precheck 静态审计 + 实施 sid_metadata + Stage3 batch 序列化 + attention-bias stub + 7 markers 验证 + close issue
**结果**: ✅ Issue #74 预检 PASS (7/7 markers 全部 PASS), 实证数据完整 (R18 强制)

---

## 1. Issue #74 跟 #71/#68/#65 同路径但实证不同分析

| 维度 | Issue #71 (Direction C 预检) | Issue #74 (Direction C 预检 sid_metadata) | 同路径? |
|------|------|------|------|
| 框架 | Stage3 vanilla T5 wrapper, 0/7 markers | Stage3 batch 携带 sid_metadata + attention-bias stub | ✅ (Stage3 vanilla T5) |
| 实施核心 | 7 markers 缺失 (sid_metadata, attention-bias stub 等) | sid_metadata schema + Stage3 batch 序列化 + attention-bias stub (7 markers 全部实施) | ❌ 不同 (Issue #74 明确给出 sid_metadata schema + attention-bias stub 实施路径) |
| Gate 1 失败机制 | 0/7 markers 实施基础缺失 | 7/7 markers 实证 PASS, 实施基础完整 | ✅ (都是 Stage3 实施) |
| 引用文献 | 未引用具体 arXiv | arXiv:2309.04082 (Curve Your Attention) | ❌ 不同 |

→ **Issue #74 跟 #71 实质同路径 (Stage3 vanilla T5 wrapper), 但 #74 给出具体 7 markers 实施路径, 实证 7/7 PASS, 不允许沿用 #71 判决**.

## 2. R11.5 决策: 立即实施 (R19 强制)

- Issue #74 描述自己给出 sid_metadata schema + attention-bias stub 实施路径
- precheck 强制执行: 7 markers 必须全部 PASS
- R19 强制: 立即实施 + 启动 GPU 验证, 不等待授权
- 实施结果: 7/7 markers 全部 PASS, 实证数据完整

## 3. Issue #74 Gate 决策 (R17/§19)

| R17 Gate | 决策 | 失败原因 |
|------|------|------|
| **Gate 1 (= Stage 1 RQ-VAE/HRQVAE)** | ⏸ STOP (Issue #74 预检任务) | Issue #74 spec 是预检任务, 不要求 Stage 1 训练 |
| **Gate 1.5 = 预检 7 markers** | ✅ **PASS** | 7/7 markers 全部 PASS: M1 schema + M2 序列化 + M3 反序列化 + M4 attention-bias stub + M5 关闭等价 + M6 开启不同 logits + M7 gradient 路径 |
| Gate 2 (= Stage 2 Sinkhorn) | ⏸ STOP | per spec |
| Gate 3 (= Stage 3 T5-mini) | ⏸ STOP | per spec |
| Gate 4 (= Stage 4 R@K eval) | ⏸ STOP | per spec |

→ **Issue #74 预检 PASS, 7 markers 实证完整**.

---

result: Issue #74 [方向C 预检 sid_metadata] 预检 7/7 markers 全部 PASS. R18 强制实证: 7 markers (M1 schema + M2 序列化 + M3 反序列化 + M4 attention-bias stub + M5 关闭等价 + M6 开启不同 logits + M7 gradient 路径) 全部 PASS. R19 强制立即实施 + 启动 GPU 验证. 跟 #71/#68/#65 同路径 Stage3 vanilla T5, 但 #74 给出 7 markers 实施路径, 实证成功. Gate 1=Stage1 ⏸ STOP (预检任务), Gate 1.5=预检 ✅ PASS, Gate 2/3/4 ⏸ STOP per spec.
