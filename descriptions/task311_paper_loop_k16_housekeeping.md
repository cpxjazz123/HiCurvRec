# Task #311 — K16 收口后 housekeeping: paper.md §6.7.4 + loop.md §16 + TASKS_INDEX 整理

**日期**: 2026-07-30
**状态**: ✅ 启动 (R9 空洞填补 + housekeeping)
**目的**: K16 (Stage 4 repetition_penalty 0pp) 收口后, 整理文档体系: paper.md §6.7.4 反映 K14/K15/K16 完整 Stage 4 inference 协议层结论, loop.md §16 清理已完成条目 (R8 强制), TASKS_INDEX.md 更新 snapshot + K16 引用.

**R11.5 自主决策**:
- R9 审计发现 #311 空洞 (task309/310 之间缺 description), 立即填补
- 零 GPU 成本 (housekeeping 仅文档操作)
- R8 强制清理 §16 中已完成条目 (task307/308/310 全部闭环, 但是否还在 §16 表格内未确认)

**关联**:
- K14 (task307): beam_size 20→50 +2.3% R@10
- K15 (task308): length_penalty 0pp
- K16 (task310): repetition_penalty 0pp
- 三者联立锁定 Stage 4 inference 协议层唯一杠杆 = beam_size

**预期**: ~15min (paper.md §6.7.4 段落级更新 + loop.md §16 表格整理 + TASKS_INDEX.md K16 引用)

**关键决策点 (R11.3 透明)**:
- 选 housekeeping 而不是新 GPU 任务: 因为 task309 (GPU 1, ~3.5h 剩余) + task312 (GPU 0, ~85min 剩余) 都在跑, GPU 2/3 留给后续 Stage 4 eval (R7 不抢 GPU)
- R9 修复优先: 防止 #311 空洞遗留 (R9-Enforce 三层防护)

---

## 执行清单

- [ ] paper.md §6.7.4 更新 K14/K15/K16 收口 (Stage 4 inference protocol = 单点 beam_size 杠杆)
- [ ] loop.md §16 表格清理 (R8: 移除已完成条目行)
- [ ] TASKS_INDEX.md 更新 snapshot date + K16 引用
- [ ] R9 audit 重跑 (确认 #311 已填, 无新空洞)
- [ ] verdict 文件落盘