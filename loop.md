# GRID Reproduction Task Plan — GitLab Issue Rules (One-Liner Subset)

> 本文件 = 11 个核心 GitLab Issue 规则, 每个一句. **不含表格 / 命令模板 / 实施细节** (按需自行实现).

---

## 1. R14 §15.6 — Issue 自动监听
每次 loop tick 第一步 `glab issue list --all` 扫描, 发现 open issue 立即按 R11.5 自主处理, commit 粒度 = 每 issue 一个独立 commit.

## 2. R15 §15.7 — Issue 闭环 push
issue 闭环 = verdict 落盘 + commit + `git push` + `glab issue close` 四件套齐全, 顺序固定 (commit 在本地 ≠ reviewer 可见).

## 3. §16 + §16.1 — 活跃任务表格
活跃任务表格是 loop 唯一允许的执行情况记录位置, 同时活跃 ≤ 5 行 (Task ID / Issue / 类型 / 阶段 / GPU / 进度 / ETA), 完成立即删除该行 (R8 强制, 不写 "已归档").

## 4. R16 §18 — 每次 tick 检查 + 关闭
每次 tick 第一步检查 open issue, 有 → 必须完成 + `glab issue close`, 没有 → 允许 idle (R10 v2 保留).

## 5. R17 §19 — commit 含 Gate + 失败原因
commit message 必含 `Gate <N> FAIL/PASS` + 失败原因, 4 Gate = Stage 1 RQ-VAE / Stage 2 Sinkhorn / Stage 3 T5-mini / Stage 4 R@K eval, 前 Gate FAIL → 后 Gate STOP.

## 6. R18 §20 — 路径差异必须做实验
新 issue 跟历史做 4 维度对比 (D1 spec 摘录 / D2 实施核心 / D3 Gate 1 失败机制 / D4 引用文献), 任何不同 → 必须做实验 (precheck / GPU 训练 / 端到端 eval), 不能凭"路径同构" NO-GO.

## 7. R19 §21 — AI 激进 owner
AI = owner, 任何决策不需要等 owner 拍板, precheck PASS → 立即启动 GPU 训练, 4 维度不一致 → 立即实施 + 实验, 跨 issue 必须并行 (一张 GPU 一个).

## 8. R20 §22 — commit + comment 详细 4 Gate
commit + issue comment 都必须详细回答 4 Gate (≥3-5 行/Gate: 状态 / 关键数据 / 失败原因 / verdict 路径 / commit hash), close issue 前必发 comment.

## 9. R21 §23 — commit hash 必须明示
commit hash 必须明示 (不允许 "pending" / "TBD" / "TODO" 占位), comment 必须在 commit + push 之后发 (拿到 hash 才能发), 顺序固定: commit+push → comment(含 hash) → close.

## 10. R22 §24 — 新 issue 立即开工
任何 OPEN issue 出现 → 立即 R16+R17+R18+R20+R21 闭环, owner 派工立即执行, 4 卡全占 → 换 GPU / nohup 后台排队 / 缩减规模 (不允许 "等已用卡释放"), 禁止 "等下一轮" / "是否启动?" / "等 owner 拍板".

## 11. R23 §25 — 明显失败立即终止
每个 tick 扫一眼活跃训练, 7 信号任一触发立即终止 (val_R@10=0 跨 ≥2 checkpoint / loss 不下降 / val loss 反向 / wrapper broken / ckpt 不存 / NaN/Inf / GPU 100% 但 loss 不变), 立即 kill -9 + NO-GO verdict + commit + push + close issue.

## 12. R24 §26 — 禁止任务暂停等待 owner 拍板
每个 tick 检查 in_progress 任务是否真在执行 (有活跃 PID + file mtime 更新), 若 in_progress 但无活跃行为 → 立即自主决策 + 立即执行, 不允许任何 "等待 owner 拍板" / "wrapper 复杂" / "等下一 tick" 话术.

---

## §16 — 活跃任务表 (R8 强制, 同时活跃 ≤ 5 行)

| Task ID | Issue | 类型 | 阶段 | GPU | 进度 | ETA |
|---------|-------|------|------|-----|------|-----|

(完成立即删除该行 — R8 强制, 不写 "已归档"; verdicts/ 保留 verdict 文件)

> **当前活跃任务**:
>
> | Task ID | Issue | 类型 | 阶段 | GPU | 进度 | ETA |
> |---------|-------|------|------|-----|------|-----|
> | #98 (in_progress) | #61 | 方向A Stage3 | T5 训练 200ep, P0 5 项加速 | GPU 0 | epoch 5/200 完成 (best NDCG@20=0.0611), 15s/epoch train + 14s eval | ~1.7 h |
> | #102 (pending) | #61 | 方向A Stage4 | R@K/NDCG eval | GPU 0 (after Stage3) | 待 Stage3 完成启动 | ~10 min |
> | #103 (pending) | #61 | 方向A verdict | 闭环 commit + push + close | - | 待 Stage4 完成 | ~5 min |
>
> 已关: #56/#57/#58/#59/#60. 4 GPU 中 GPU 0 占 (Stage3 200ep + P0 5 项加速), 1-3 空闲.
>
> **Stage3 三轮加速累积 (用户指示 2026-08-06)**:
> - L1: BATCH_SIZE 256→1024 + EVAL_INTERVAL=5 (DECOR 对齐)
> - L2: P0 5 项框架加速 — NUM_WORKERS=4 + pin_memory + persistent_workers + TF32 + fused AdamW + torch.compile
> - L1+L2 综合: 96s/epoch → 15s train + 14s eval = 29s/effective epoch (3.3× 加速), 5.3h → 1.7h
>
> Stage3 PID=2231158 (active, GPU 0 93%/16.3GB). HG-Rec 上游 DataLoader 不允许改 → 用 `_FastGenRecDataLoader` 子类注入 pin_memory/persistent_workers.