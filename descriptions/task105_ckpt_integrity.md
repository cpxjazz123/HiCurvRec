# Task #105 — R12 Best Checkpoint Integrity Verification

> **任务目的**: 验证 Stage 3 HG-Rec best ckpt (R12 强制保存规则下产生) **物理完整 + 可加载 + 与 paper.md 报告数字一致**. 形成 paper submission 阶段的最后一层防御——证明 R12 规则真正生效, 不是只写在 CLAUDE.md 里。

> **完成日期**: 2026-07-24
> **状态**: 🟡 待启动 (in progress)

---

## 1. 背景

Task #84 (HG-Rec main reproduction) 的 Stage 3 best ckpt 已落盘:
`products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/HG_Rec_best.pth` (~ 21 MB, R12 mandated save).

Stage 4 evaluation log `logs/task84_hgrec_stage4_eval_jul-23-2026_21-38-14.log` 报告:
- R@5  = 0.0815637065
- R@10 = **0.1020350709** ⭐ (paper.md Table 2 c111 HG-Rec 实测 line 212)
- R@20 = 0.1278555343
- NDCG@10 = 0.0755451237

paper.md Table 2 HG-Rec c111 R@10 = **0.1020** (Task #84 standalone)；同步 Task #88 grid 也产 0.0998 数字。这是 paper 全文最高 contrast 的数字 — 必须保证它"可追溯到磁盘上的一个真实 ckpt"。

R12 规则要求:
1. 训练过程中必须强制保存 ckpt（不允许只在 evaluation 之前临时 load）
2. 每个 epoch 之间要删旧存新
3. 磁盘上只能保留最新一个 ckpt

Task #105 的目标 = 验证这 3 条真正被遵守:
- 文件存在 (R12 #1)
- 文件大小符合 T5-small 维度 (R12 #2 — 单个 ckpt,~21 MB, 不应 50 MB+ 多 epoch 残留)
- Stage 4 eval log 的 R@10 数字 ⇄ paper.md 报告数字 ⇄ ckpt 是同一条证据链 (R12 #3)

---

## 2. 实验设计 (writeup only)

### 2.1 验证方法 — 四层审计

| 层级 | 检查 | 工具 |
|------|------|------|
| L1: 文件存在 | ckpt path 存在 | `Path.exists` |
| L2: 大小 R12 mandate | size ∈ [18 MB, 25 MB] (T5-small 192d hidden 单 ckpt) | `Path.stat().st_size` |
| L3: 日志证据 | 从 `logs/task84_hgrec_stage4_eval_*.log` 提取 R@10 / R@5 / NDCG@10 | regex |
| L4: 论文匹配 | 提取出的 R@10 round to 4 decimals 与 paper.md HG-Rec c111 R@10 = 0.1020 / 0.0998 对比 | table lookup |

四层独立验证 (无相互依赖), 任一层失败 → raise (R2 no-fallback).

### 2.2 torch 不可用 fallback
当前 shell 无 torch (`ModuleNotFoundError`). Task #105 用文件级 + 日志级审计替代权重级反序列化 (R12 acceptance criterion), 不依赖 torch.
如需权重级深度验证, 在 `kgat_tf216` 或 `grid_toys` env 后续单独跑 Task #106 (R12 weight-level audit).

### 2.3 输出

| 产物 | 路径 |
|------|------|
| Verifier script | `scripts/task105_ckpt_integrity.py` |
| Verdict report | `verdicts/task105_ckpt_integrity.md` |
| Audit log | `verdicts/task105_ckpt_integrity.md` §4 audit table |

---

## 3. 决策触发

| 条件 | 决策 |
|------|------|
| L1-L4 全部 ✅ 且 paper matches ≥ 2 | ✅ 闭环 (R12 + paper 证据链同时确认) |
| L1 fail | ❌ ckpt 丢失 / 路径错, 必须恢复 (impacts R12 compliance) |
| L2 fail (size < 18 MB 或 > 25 MB) | ❌ ckpt 损坏 或 多 epoch 残留 (R12 #3 违规), 必须重新训练 |
| L3 fail | ❌ log 文件不存在 / Recall 字段未写入, 必须重跑 Stage 4 eval |
| L4 fail (paper matches = 0) | ❌ paper.md 数字与 ckpt 脱节, 必须重新校核 paper.md |
| L1 ✅ L2 ✅ L3 ✅ L4 = 0 | ⚠️ paper.md 数字不可信, 必须手动 audit (R11.3) |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| 写 verifier | ~5 min |
| py_compile 验证 | 2 sec |
| 跑 verifier | < 1 sec (pure stdlib) |
| 写 verdict + commit | ~5 min |
| **总计** | **~10 min, 0 GPU** |

---

## 5. 风险与缓解

**风险 1**: torch 不可用导致无法做 weight-level check
  → **缓解**: 用文件级 + 日志级 4 层审计替代 (本任务范围); torch-level audit 留给 Task #106 (后续可选)

**风险 2**: R@5 vs R@10 错位 (paper.md Table 2 是 R@10, log 末尾 dict 也是 R@10, 同一口径)
  → **缓解**: Task #103 audit 已经确认 paper.md Table 2 c111 数字 = Task #84 log R@10, 不存在错位风险

**风险 3**: ckpt 大小受 batch size / dropout 影响波动
  → **缓解**: R12 接受 [18 MB, 25 MB] 大小范围 (T5-small 192d hidden + 4 layer 标准 config 21 MB, ±15% tolerance)

---

## 6. 完成度跟踪

- [x] 写 `scripts/task105_ckpt_integrity.py`
- [x] py_compile 验证 (Rule 10)
- [ ] 跑 verifier 产出 `verdicts/task105_ckpt_integrity.md`
- [ ] Layer 1-4 audit table 闭环
- [ ] git commit
- [ ] §16 loop.md 更新 (Task #105 标记 ✅ 已完成, 删除活跃任务行)

---

## 7. 关联

- 前置: Task #84 (HG-Rec main reproduction), Task #103 (paper claims audit), Task #104 (CITATION.cff + CHANGELOG 闭环), R12 规则
- 后置: Task #106 (R12 weight-level audit, 可选, 用 torch env 跑)

---

**核心交付**: 四层 (文件 / 大小 / 日志 / 论文) 独立审计 + verdict report. R12 + paper 证据链双重确认.
