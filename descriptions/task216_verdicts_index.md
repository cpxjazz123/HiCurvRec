# Task #216 — verdicts/index.md 整理 + paper §6 future work 收尾

**日期**: 2026-07-26
**父级**: Task #215 paper §4 收尾完成 → housekeeping
**优先级**: 🟢 低 (content creation, 0 卡, 但有助 reader 导航)

---

## 1. 背景

Task #193-#215 共 23 个 task 全部 verdict 已写 (verdicts/task{N}_*.md). 但没有统一 index 文件, reader 想找"哪个 task 给我 R@10=0.0816" 或 "哪个 task 验证 Sinkhorn" 不容易快速定位.

## 2. 目标 (1-2h, 0 卡)

1. **写 verdicts/index.md**: 按主题分类所有 task verdict, 表格化 + 短描述
2. **更新 papers/paper.md §6 future work**: 替换原"未来可能方向" 为"7 方向 NO-GO 总结", 让 paper self-contained

## 3. 实施步骤

1. 0.5h: 列出所有 verdicts/task{N}_*.md (按数字顺序)
2. 0.5h: 按主题分组:
   - 主题 A: Baseline 复现 (Task #84, #143, #144)
   - 主题 B: 几何变体 NO-GO (Task #199/200/201/203/208/209/211/212/213/214)
   - 主题 C: κ 调试 (Task #164/165/162/166/167/168/169/170)
   - 主题 D: T5 容量 / 训练 (Task #157/158/159/160/161)
   - 主题 E: 复现审计 (Task #82/88/89/90/91/117/118/207)
   - 主题 F: paper 收尾 (Task #30/215)
3. 0.5h: 写 verdicts/index.md 表格
4. 0.5h: 更新 paper.md §6 future work

## 4. 交付物

- verdicts/index.md (新文件, 表格化)
- papers/paper.md §6 future work 更新

## 5. 链接

- 所有 verdict: verdicts/task{N}_*.md (Task #30-#215)
- Paper 主文件: papers/paper.md
