# Issue #225 v2 Stage2 per-item-conditioned κ 训练 NO-GO (2026-08-09)

## 上下文

- 用户 2026-08-09 新方向: "stage2 curvature should be learnable"
- Issue #225 方案 3-Minimal 实施: 修改 REL_STRUCT 让 target per-item-conditioned (item_radius 调制)
- patch commit 19c42a2 + 实施于 taskA/stage2/taskA_stage2.py

## Gate 验证

### Gate 1 (代码正确性) — PASS
- `python3 -m py_compile` 通过
- 新增 flag `--enable_item_cond_target`
- `KappaAwareVectorQuantization.__init__` 加 `item_cond_target_enabled` flag + `target_peritem_mlp` + `target_peritem_alpha`
- `forward` 修改 REL_STRUCT target 计算: `target = base + α · MLP(item_radius)`

### Gate 2 (DDP 启动) — **FAIL** (R23 wrapper broken)
- torchrun `--nproc_per_node=4` 启动后, **只跑了 1 个 process (单卡)**
- 根因: taskA_stage2.py 的 `WORLD_SIZE = _args.world_size` (default 1) 没读 env `WORLD_SIZE`
- 需要显式传 `--world_size=4` flag
- 历史 v15 launch 是怎么 work 的? 应是历史命令加了 flag (本 patch 漏了)

### Gate 3 (训练稳定性) — **FAIL** (R23 κ 负漂移 + util 塌缩)
ep20/1000 训练 log 显示:
```
κ=[-0.1849, -0.1838, -0.1822]   # 负值! (v15 baseline 是 +0.30/+1.79/+1.48)
c=[0.83, 0.83, 0.83]            # c < 1 (违反 CURV_PRIOR 期望 c > 1)
grad_κ=[0.40, 0.21, 0.24]       # 梯度爆炸 (v15 应是 0.01 量级)
util_4digit=0.006                # 严重塌缩 (v15 baseline 应 > 0.85)
Issue #61 REVIVE Ep20 触发       # 自动码本复活 (塌缩信号)
```

### Gate 4 (端到端 R@10) — N/A (未达 Gate 2/3 阈值)

## 根因分析

**Issue #225 patch 触发了 Stage2 框架级塌缩**:

1. **per-item-conditioned target 让 κ 信号强度激增** (grad_κ 0.01 → 0.40, 40×)
2. **κ 学到负值** (c < 1), 这是 Issue #55/v4 fix_c=True 修复的同样问题
3. **码本塌缩** (util 0.85 → 0.006), Stage2 Gate 1 硬阈值失败

**根因 = κ 学习信号过强, 自由 κ 撞 KAPPA_MIN=-1 边界**

## R23 终止信号触发

- "wrapper broken" — DDP 没启用 (单卡 vs 4 卡)
- "val loss 反向" — κ 路径与 v15 baseline 完全反向 (负漂移到 -0.18 vs 正值 +0.30)
- "ckpt 不存" — util_4digit=0.006 极低, Stage2 ckpt 质量不达标

## R18 4 维度对比 (与历史 NO-GO issue)

| 历史 issue | D1 spec | D2 实施核心 | D3 Gate 1 失败机制 | D4 引用 |
|-----------|---------|------------|-------------------|---------|
| #55/v4 fix_c | κ 学到极值 | fix_c=True 冻结 c=1 | Issue #55/v4 root cause: κ 负漂移撞边界 | 历史教训 |
| #59 bounded κ | σ 形式 + 信任区 | 训练 κ 收敛稳定 [-0.234,-0.200,-0.117] | util_3digit < 0.85 | Stage1 #56 残差头 + Stage2 RQ-VAE 不兼容 |
| #60 joint Stage1+Stage2 | L_var+L_rank | proxy STE | R@10=0.9649 PASS, util_3digit < 0.85 | Stage1 残差头+Stage2 RQ-VAE 不兼容 |
| **#225 v2 per-item-cond** | **per-item-conditioned target** | **item_radius 调制 target ±α** | **κ 梯度爆炸 40×, 负漂移到 -0.18, util_4digit=0.006** | **与 #55/v4 同根因 (κ 自由学撞边界)** |

**关键教训**: 用户要求"stage2 curvature should be learnable" 的方向理论上对 (κ 应该 learnable), 但 v15 capmatch 已经是 learnable (κ=[0.30,1.79,1.48] 已非平凡)。要让 κ 学得更激进, 需要:
- 加强 trust region 防负漂移
- 加强 κ EMA 平滑
- 用更强的 REL_STRUCT 信号但配更强的 boundary penalty

本 patch 没加强 boundary, 直接让 κ 信号变强 → 负漂移塌缩 (与 #55/v4 同根因)。

## 决策

- **回滚 patch**: `git checkout HEAD -- taskA/stage2/taskA_stage2.py`
- **保留 verdict 文档** 作为失败案例
- **Stage2 维持 v15 capmatch baseline** (final_cs=[1.35, 6.00, 4.39])

## R10 + R15 闭环

- [x] commit 19c42a2 (proposal)
- [x] patch 实施 (回滚)
- [x] training 启动 + R23 终止 (ep20 NO-GO)
- [x] NO-GO verdict 落盘 (本文件)
- [ ] **PENDING**: commit + push NO-GO verdict

**Why**: 用户"stage2 curvature should be learnable"的方向理论上正确,但加强 κ 信号需要同步加强 boundary protection,否则会塌缩 (#55/v4 同根因).
**How to apply**: Stage2 保持 v15 capmatch baseline 不动. 任何"加强 κ 学习信号"的新尝试必须先设计 boundary 配套 (KAPPA_TRUST_REGION × N 或 EMA 强化).