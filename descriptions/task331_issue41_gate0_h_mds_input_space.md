# Task #331 — Issue #41 Gate 0 — Sala 2018 h-MDS 在输入空间 (768d) 而非 residual 空间

**日期**: 2026-07-30 13:27
**触发**: Issue #41 owner 13:25 创建, R14 强制处理 (零 GPU 纯计算, 立即执行)
**状态**: 🔄 Gate 0 PENDING (本 description 创建后立即跑)

---

## 1. Issue #41 核心主张

| 项 | 内容 |
|----|------|
| 主方法 | Sala, De Sa, Gu, Ré (2018) "Representation Tradeoffs for Hyperbolic Embeddings" (ICML 2018, arXiv:1804.03329) h-MDS / 失真-曲率-维度权衡 |
| 关键差异 | 测 **输入空间** 768d (非 residual 空间), 验证真双曲信号是否仍存在 |
| 跨验证 | Task #70 Ollivier κ=-0.65~-0.84 + Task #80 residual κ≈0 |
| 新意 | 输入空间 + h-MDS 此前从未在本仓库使用过, 可能绕过 "residual 空间已被 encoder 压平" 这条贯穿 #1-#40 的共同瓶颈 |

## 2. Gate 0 实验设计 (Issue #41 硬性前置, 零 GPU)

### 2.1 (a) 输入空间 h-MDS 估计

**自变量**: 768-dim 原始输入 embedding (Musical_Instruments, 9922 items)
**方法**: 对每个候选 κ ∈ κ_grid, 用 Riemannian GD 在 Poincaré ball 做 MDS, Kruskal stress-1 衡量拟合质量
**κ_grid**: `[0.0, -0.05, -0.1, -0.2, -0.5, -1.0, -2.0]` (覆盖 0 ~ 强双曲)
**通过条件**: 最优 κ 应显著偏离 0 (|κ| > 0.05) 且与 Task #70 Ollivier (负) 方向一致

### 2.2 (b) Residual 空间对照组 (sanity check)

**自变量**: Task #84 baseline ckpt 4 层 residual 点云 (L0/L1/L2/L3, 32d)
**方法**: 复用 `task80_stage1c_true_distortion.compute_true_distortion` + 相同 κ_grid
**通过条件**: 最优 κ 应 ≈ 0 (复现 Task #80 既有结论)
**硬停止**: 若对照组复现失败 → STOP 排查实现差异, 不继续 Gate 1

### 2.3 H2 类目树交叉验证 (可选)

**自变量**: `category_depth.npy` (N=9922, depth 2-16, ρ range 1.5-2.9)
**方法**: Sala 2018 树状组合构造法: 树深度 d → ρ_d = log(d) / sqrt(-κ), 反推 κ
**通过条件**: 估出的 κ 应与 (a) 同方向、量级可比

## 3. 实施路径 (R10 + R11.5 自主决策)

1. ✅ 创建本 description (R9 max+1 = 331)
2. ⏭️ 写 3 个 Gate 0 脚本:
   - `scripts/task331_issue41_gate0_input_h_mds.py` (a)
   - `scripts/task331_issue41_gate0_residual_control.py` (b)
   - `scripts/task331_issue41_gate0_h2_category_tree.py` (H2)
3. ⏭️ 跑 (b) 先: 复现 Task #80 κ≈0 → sanity check PASS
4. ⏭️ 跑 (a): 输入空间最优 κ 估计
5. ⏭️ 跑 H2 (若有富余时间): 树状组合法 κ 估计
6. ⏭️ verdict 落盘: `verdicts/task331_issue41_gate0_h_mds_input_space.md`
7. ⏭️ GitHub Issue #41 评论 (Stage 1 设计 only if Gate 0 PASS, 否则关闭)

## 4. 不申请 GPU (R7 + R10 + Issue #41 明确)

Gate 0 纯 CPU 计算, 跟当前 4 GPU 占用不冲突 (task327 Stage 3 + task328 R-Drop 3-arm + α=4.0 watcher + Gate 1 Stage 3+4 wait)。

## 5. 跟当前活跃任务的兼容性

- task327 Stage 3 (GPU 1 Ep125/200): 不受影响
- task328 R-Drop α=0.5/1.0/2.0 (GPU 0/2/3 Ep7/200): 不受影响
- task328 R-Drop α=4.0 watcher: 不受影响
- Gate 1 Stage 3+4 wait PID 1878613: 不受影响

## 6. 关联

- Issue #41 (主)
- Task #70 verdict: Ollivier κ=-0.65~-0.84 强双曲
- Task #80 verdict: residual 空间 κ≈0 (Idea 1 否证)
- Task #82 verdict: |κ|≈0.05 弱信号探测下限
- `scripts/task80_stage1c_true_distortion.py` (复用 compute_true_distortion)
- `scripts/task116_hgrec_delta_per_layer.py` (复用 load_hrqvae + extract_per_layer_residuals)
- `HG-Rec/dataset/Instruments/item_emb.parquet` (768d 输入)
- `HG-Rec/dataset/Instruments/category_depth.npy` (类目树)

---

result: Task #331 description 创建. Gate 0 脚本与运行待启动.