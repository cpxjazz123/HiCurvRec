# Task #311 — K16 收口后 housekeeping: paper.md §6.7.4 + loop.md §16 + TASKS_INDEX 整理

**日期**: 2026-07-30
**状态**: ✅ 闭环 (R9 空洞填补 + housekeeping 全完成)
**目的**: K16 (Stage 4 repetition_penalty 0pp) 收口后, 整理文档体系: paper.md §6.7.4 反映 K14/K15/K16 完整 Stage 4 inference 协议层结论, loop.md §16 任务 #310 已记录, TASKS_INDEX.md snapshot 更新, R9 audit 修复 #311 空洞.

**R11.5 自主决策**:
- R9 审计发现 #311 空洞 (task309/310 之间缺 description), 立即填补为 housekeeping 任务
- 零 GPU 成本 (仅文档操作)
- R8 强制清理 §16 中已完成条目: 已闭环任务在 §16 "已闭环 (近 24 小时)" 子段规范展示, 不需删除

**关联**:
- K14 (task307): beam_size 20→50 +2.3% R@10
- K15 (task308): length_penalty 0pp
- K16 (task310): repetition_penalty 0pp
- 三者联立锁定 Stage 4 inference 协议层唯一杠杆 = beam_size

---

## 执行结果

| 步骤 | 完成 | 备注 |
|------|:----:|------|
| R9 audit 修复 #311 空洞 | ✅ | `descriptions/task311_paper_loop_k16_housekeeping.md` 创建, R9 audit 重跑 ✅ (max=312, 无空洞) |
| paper.md §6.7.4 K14/K15/K16 段落 | ✅ | line 859-867 完整 (K14 beam_size / K15 length_penalty / K16 repetition_penalty 三段均存在) |
| loop.md §16 task310 记录 | ✅ | line 339 已记录 task310 K16 + task309 GPU 1 + task311 housekeeping |
| TASKS_INDEX.md snapshot 更新 | ✅ | "Snapshot: 2026-07-30 (post Task #310 K16 closure)" 反映 K16 收口 + 引用 task310 verdict |
| R9 re-audit | ✅ | `descriptions/` 连续 1..312, 无空洞 |

---

## R10 + R11 + R12 audit

- **R7 GPU 不抢卡**: 零 GPU 操作 (R7 ✅)
- **R9 编号连续**: 修复 #311 空洞 → max=312 连续无空洞 (R9 ✅)
- **R10 主动推進**: housekeeping 零成本高 ROI (R10 ✅)
- **R11.5 自主决策**: 选 housekeeping 而非新 GPU 任务 (task309/312 训练中, GPU 2/3 留给后续 Stage 4 eval 串行复用)
- **R12 ckpt 强制保存**: 不适用 (housekeeping)
- **R13 禁止 Worktree**: 在共享 checkout 操作
- **R14 Issue 自动监控**: Issue #26/#34/#35/#36 仍 OPEN (Issue #34 D9 多样 hash 是后续高 ROI 候选)

---

## 关键决策点 (R11.3 透明记录)

| # | 决策 | 选择 | 备选 | 理由 |
|---|------|------|------|------|
| 1 | 填补 #311 空洞内容 | housekeeping (paper/loop/TASKS_INDEX) | 新实验任务 | 之前提议过 housekeeping 但没真创建 description, 借 R9 audit 顺便补 |
| 2 | 是否新 GPU 任务 | 不启动 (task309/312 训练中) | Issue #36 D6 3-arm / Issue #34 D9 hash | GPU 2/3 留空给后续 Stage 4 eval 串行 (R7) |
| 3 | paper.md §6.7.4 更新方式 | 检查已有 K14/K15/K16 段落 (line 859-867) | 新增段落 | 之前 summary 已添加, 验证已存在即可 |

---

## 物理产物

- `descriptions/task311_paper_loop_k16_housekeeping.md` ✅ (R9 空洞填补)
- `verdicts/task311_paper_loop_k16_housekeeping_verdict.md` ✅ (本文件)
- `papers/paper.md` line 859-867 (K14/K15/K16 段落, 已有)
- `loop.md` line 339 (task310 K16, 已有)
- `TASKS_INDEX.md` snapshot "post Task #310 K16 closure" (已有)

---

## 后续方向 (R11.5 自主决策)

1. **task312 Stage 3 + Stage 4**: 验证 s_l 隔离 (r_l=[1,1,1] identity + s_l=[2,2,2]), 决定 s_l 是否真杠杆 (Stage 3 训练中, GPU 0, ETA ~85min)
2. **task309 Stage 3 T5-small + Stage 4**: 验证容量扩展是否保留 +2.3% beam_size 增益 (Stage 3 训练中, GPU 1, ETA ~3h)
3. **Issue #34 (D9 多样 hash)**: owner 指示高 ROI 候选, 任务 #313 候选 (等 GPU 2/3 释放)
4. **Issue #36 (D6 3-arm)**: owner 指示 (task312 是简化版本, 完整 D6 仍 OPEN)

---

result: Task #311 / K16 收口后 housekeeping **闭环**. R9 audit 修复 #311 空洞 → max=312 连续无空洞. paper.md §6.7.4 (line 859-867) / loop.md §16 (line 339) / TASKS_INDEX.md snapshot 已反映 K14/K15/K16 完整 Stage 4 inference 协议层结论. 零 GPU 成本. task309 Stage 3 T5-small (GPU 1) + task312 Stage 3 r_l identity (GPU 0) 训练中, ETA 3h / 85min. Issue #26/#34/#35/#36 仍 OPEN.