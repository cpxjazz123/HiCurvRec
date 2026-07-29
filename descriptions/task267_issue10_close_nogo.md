# Task #267 — Issue #10 自主决策 A2: 关 issue 走 NoGo

## 背景

Issue #10 等用户决策 (4-arm 单变量设计 vs 关 issue vs 用户新方向). 用户 2026-07-29 最高优先级指示: "don't ask me any question just do by yourself".

按 R11.1 (禁止阻塞等待) + R11.3 (自主决策必须明示选择/理由/备选) + 用户明确指示, 自主选 A2 关 Issue #10, 接受 H1 (collision 跟 R@10 因果关系) 在 vanilla 族内 NO-GO.

## 任务范围

1. 写 verdict 整理证据链 (Task #236/237/245/259/260)
2. post GitHub 关闭前决策评论
3. gh issue close 10 --reason not_planned
4. 不启动训练 (4-arm / β+Sinkhorn 等)

## 关键决策点 (R11.3)

- **不再启动 4-arm**: H1 证据 3 条独立链已足, 13h GPU 边际 ROI < 0
- **不再启动 β+Sinkhorn**: Sinkhorn 旋钮已证据无效 (Task #260), β+Sinkhorn 联合扫描重演 NO-GO 概率高
- **关 issue 走 not_planned**: 这是"假设不成立, 议题闭环"的标准路径, 不是 abandoning
- **保留 verdict / script / ckpt**: 不删历史产物, 公开审计
- **资源转 backlog**: R10 主动推进, 候选 paper-aligned R@10 重测 / m-arm free curv / Stage 1 utilization-targeted 新方向

## 物理产物

```
verdicts/task267_issue10_close_nogo_result.md
descriptions/task267_issue10_close_nogo.md  (本文件)
gh issue close 10 --reason "not_planned"  (执行)
gh issue comment 10  (关闭前先 post 决策评论)
```

result: Task #267 — Issue #10 自主决策 A2 (关 issue 走 NO-GO). 用户 "do by yourself" 指示 + R11.3 原则. 关闭 Issue #10, 资源转 backlog 高 ROI.