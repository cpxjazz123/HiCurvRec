# Task #419 / Issue #126 [方向C Gate4] 固定 Gate3-PASS product-stereographic adapter 的 Task84 统一测试

**日期**: 2026-08-01
**Issue**: #126 [方向C Gate4] — 用 #123 adapter checkpoint + Task84 HG-Rec control 在同一 evaluator 跑 R@5/10/20, NDCG@5/10/20
**任务**:
1. **复用 #123 adapter checkpoint** + Task84 HG-Rec ckpt
2. **Musical_Instruments, seed=42, 相同 split, beam=20, SIDRetrievalEvaluator** (per Issue spec §Gate4)
3. **同一测试入口**: adapter checkpoint + HG-Rec control
4. **完整输出**:
   - R@5, R@10, R@20
   - NDCG@5, NDCG@10, NDCG@20
   - checkpoint/SHA256, 配置, 样本数
   - 原始 evaluator 日志
5. **复跑一次** (per spec §Gate4 3): 同配置跑两次, 报告每项差值; 不一致则 FAIL/PENDING

**Gate 4 PASS/Target reached 唯一条件**:
- 实际、可复核的 test R@10 > 0.1020
- 六项指标齐全 + 复跑证据
- 相对 Task84 baseline (0.0816/0.1020/0.1279/0.0690/0.0755/0.0821) 报告差值

**前置**:
- 复用 task84 ckpt (56d046db...) + task396 SID (9773e96a...) + #123 adapter checkpoint
- 禁止 global κ, fixed-only, pure Euclidean bypass, 替换数据集, 修改 evaluator, 另建模型
- 不重训 adapter, 不改 κ/SID/训练协议

**结果**: ⏸ 进行中 (R22 + R19 立即开工, GPU 2)

---

## R17 + R20 强制 4 Gate 详细内容

### Gate 1/2/3 复核 (per spec §复核):
- Gate 1 (Stage 1): task84 HG_Rec_best ckpt SHA256: `56d046dbabdb1930691f1361419b030b6912409853fe84e645c67072ffecb86e` ✅ PASS
- Gate 2 (Stage 2): task396 SID SHA256: `9773e96a57fad9323ed8a37b99d3d3eb5cff5e7ac40895ccd90fcc5d828537b8` shape (9922,4) unique 9922/9922 ✅ PASS
- Gate 3 (Stage 3): #123 dual-gate adapter 6/6 check PASS (verdict task416)

### Gate 4 = Stage 4: 本任务核心 (Task84 统一测试)
- Adapter checkpoint + HG-Rec control
- R@5/10/20, NDCG@5/10/20
- 双复跑验证

---

## 跨方向联立 (R18 实证 4 维度)

| 维度 | Issue #123 (task416, closed PASS) | Issue #126 (本 task) |
|------|-----------------------------------|----------------------|
| **D1 spec 摘录** | Gate 3 双态 gate architecture | **Gate 4 复用 #123 ckpt 跑 R@K eval** ✅ |
| **D2 实施核心** | zero+active dual gate 30 epoch 训练 | **统一 evaluator 跑 R@5/10/20 + NDCG@5/10/20** ✅ |
| **D3 Gate 失败机制** | (Gate 3 已 PASS, 无失败) | **test R@10 > 0.1020 才 Target reached** ✅ |
| **D4 引用文献** | arXiv:2309.04082 | arXiv:2309.04082 ✅ |

**R18 v2 强制结论**: Issue #126 跟 #123 路径**有差异** (Stage 4 eval vs Stage 3 architecture). 必须做新实验 (eval).

---

## R11.5 自主决策记录

| 决策点 | 选择 | 理由 |
|--------|------|------|
| Adapter checkpoint | 复用 #123 task416 ckpt | spec §Gate4 1 强制 (固定 Gate3-PASS ckpt) |
| Evaluator | SIDRetrievalEvaluator + beam=20 | Task84 同协议 |
| 数据集 | Musical_Instruments + Task84 相同 split | spec §Gate4 1 强制 |
| Seed | 42 | spec §Gate4 1 强制 |
| 复跑次数 | 2 次 (per spec §Gate4 3) | spec 强制双复跑 |
| GPU 分配 | GPU 2 | R7 + R19 跨 issue 并行 |
| Conda env | `genrec_env` (R1 默认) | R1 v1 通用 GPU/CPU 计算任务 |
| Gate 4 决策 | TBD per R@10 数值 | Target reached 仅 R@10>0.1020 |

---

## 后续 (per R22 + R19 + R16)

1. ✅ 写 description (本文件)
2. ⏳ 写 task419_issue126_stage4_eval.py (Stage 4 eval pipeline)
3. ⏳ 跑 Stage 4 eval (双复跑)
4. ⏳ 写 Gate 4 verdict + commit + push + close Issue #126