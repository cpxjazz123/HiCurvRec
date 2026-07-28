# Task #256 — Issue #10 方向 A 中点配置: max_sinkhorn_iters=20 完整 launcher

## 背景

Issue #10 (3-arm converged collision) 仍 open (Gate 1 PARTIAL FAIL + 等 R11.4 用户决策).
Task #255 (dry-run) 汇总 3-arm Sinkhorn 强度曲线 (max_iters ∈ {0, 10, 30}), 观察 U 形:
- 0 → 10: collision 0.99 → 0.10, R@10 +0.0001 (几乎不动)
- 10 → 30: collision 0.10 → 0.05, R@10 +0.0038 (杠杆区域)

**需要 max_iters=20 中点配置** 验证 10-20-30 之间 collision 曲线是否单调.

## 任务范围

写 Task #256 完整 launcher (Stage 2/3/4 全套), **不启动**, 等用户决策 Issue #10 方向 A 后立即可跑.

R11.3 决策: 0 GPU 决策点配置 (launcher 写完), R11.4 不可逆决策点 (启动 Stage 3 训练 50 ep) 等用户决策.

## 关键决策点 (R11.3)

- **跟 Task #237 (max_iters=10) 平行**: 同 vanilla #84 Stage 1 codebook, 同 Stage 3 配置, 只改 max_sinkhorn_iters=20
- **R12 ckpt 强制存**: save_strategy=epoch + save_total_limit=1 (latest ckpt only)
- **R88 daemon**: PID 文件写 $PROD_DIR/_TRAINING_PID
- **GPU 0**: 4 卡空闲, 按 R7 选完全空闲卡
- **不抢 GPU**: heartbeat + R12 都做好, 训练意外中断可恢复

## 物理产物

```
scripts/task256_issue10_armB_max20_full_chain.sh
```

(无 description/verdict 是因为本任务本身是 Issue #10 决策前的预准备, 等用户决策 A 后立即跑, 跑完写 verdict)

## 未来 (等用户决策)

- Issue #10 方向 A 通过 → `bash scripts/task256_issue10_armB_max20_full_chain.sh`
- 50 epoch 训练 ~30 min, Stage 4 eval ~1 min
- 写 verdict `verdicts/task256_issue10_armB_max20_result.md` 含 3-arm 曲线 (0/10/20/30)
- 跟 Task #237/Task #255 合并 commit

## 风险

- Task #84 #84 ckpt 路径可能已移位 (Stage 1 ckpt 路径是 Jul-23-2026 20-08-06 — 较早). 若 ckpt 不存在, Stage 2 直接 fail, 需要先回 task84 重跑 Stage 1 推断
- 跟 Task #237 一样的 5cond safety check (R12 ckpt, R88 heartbeat)

result: Task #256 — Issue #10 方向 A 中点配置 (max_iters=20) 完整 launcher 预准备完成. R11.4 不可逆决策等用户. 启动命令即 `bash scripts/task256_issue10_armB_max20_full_chain.sh`.
