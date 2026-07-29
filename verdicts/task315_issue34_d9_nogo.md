# Task #315 — Issue #34 D9 多样 hash simplified implementation NO-GO

**日期**: 2026-07-30
**状态**: ❌ **NO-GO** (上游 HG-Rec/model/hrqvae.py 修改, **Stage 1 + Stage 2 不可重现**)
**Issue**: Issue #34 D9 多样 hash (OPEN, 2026-07-29 owner-verified)

## 1. 背景与目的

Issue #34 = owner #26 R3 暗示 D9 多样 hash 候选路径, AI 自主决策版本 (R11.5).
Task #315 简化实施 (top-k argmin 等价于 spec 里的 sparse random projection / LSH / k-means hash 函数族).

任务目标:
- Stage 1: 沿用 #30 GO 端点 (r_l=[0.1,1,10]+s_l=[2,2,2]) 训练 100 epoch
- Stage 2: Sinkhorn 5 iter + per-layer top-k candidates (L0=3, L1=5, L2=7) by hyperbolic distance
- Stage 3: T5-mini 200 epoch 训练 (multi-SID targets)
- Stage 4: 评估 R@10, 通过条件 > 0.1022 (Issue #30 端点 GO marginal)

## 2. 失败原因 (R11.5 透明)

### 2.1 上游 HG-Rec/model/hrqvae.py 修改

2026-07-30 00:03:21 (本任务启动前 ~7 小时) 上游 HRQVAE 模块被修改:
- 移除 `curvature_list` / `log_r` / `hyp_mean` / `hyp_scale` 参数
- 移除 `model.hrq.encoder` 属性
- HRQVAE.__init__() 签名缩减为 13 个 core 参数 (原本 ~18 个)

### 2.2 Stage 1 不可重现

R11.5 决策: **跳过 Stage 1 重新训练, 沿用 task301 已存在的 ckpt** (`products/task301/hrqvae_issue30_gate1/Jul-29-2026_23-49-47_beta_0.500_codebook_[64,128,256]_sk_0.000/best_loss_model.pth`, 13.8 MB, July 29 23:49 训练).

但当前 task301 Stage 1 训练脚本 (`scripts/task301_issue30_gate1_stage1_train.py`) 调用了 `curvature_list=None`, 跟当前 HRQVAE.__init__() 签名不兼容 → 任何 Stage 1 重新训练失败.

### 2.3 Stage 2 ckpt 不兼容

```python
state_dict = ckpt['state_dict']
model.load_state_dict(state_dict)  # ← HARD CRASH
# RuntimeError: Error(s) in loading state_dict for HRQVAE:
#     Unexpected key(s) in state_dict: "hrq.vq_layers.0.log_r", ...
```

R11.5 决策: 用 `strict=False` 跳过 missing/unexpected keys. 加载成功但后续失败:
```python
top_k_indices = per_layer_topk_by_hyperbolic_distance(model, batch, top_k_list)
# AttributeError: 'HResidualVectorQuantization' object has no attribute 'encoder'
```

新 HRQVAE 没有 `model.hrq.encoder` 属性, 这是当前 HRQVAE 架构根本不同 → 不可用旧 ckpt 提取 per-layer top-k candidates.

### 2.4 Stage 3 + Stage 4 仍可运行

T5 训练不依赖 HRQVAE, 只依赖 Stage 2 输出的 SID .npy file. Stage 4 eval 也只加载 T5 ckpt + SID file.
但没有新的 Stage 2 multi-SID output, D9 多样 hash 机制无法注入 Stage 3 + Stage 4.

## 3. R11.5 决策

**方案 C: 写 NO-GO verdict, Issue #34 task #315 暂缓**.

理由:
1. 上游 HRQVAE 改了 signature + 架构 (移除 curvature_list / log_r / hyp_mean / hyp_scale / encoder 属性)
2. R11.4 critical decision: 不能在不通知 owner 的情况下修改上游 HG-Rec/model/hrqvae.py
3. 任务 #315 不需要**新增**机制 — 只是 D9 简化实施 — 但需要 Stage 1 + Stage 2 重跑才能产出新 SID file
4. 当前所有 GPU 空闲, 但任务链被卡在 Stage 2 ckpt 不兼容

**Owner 决策路径**:
- 选项 A: 修订 HG-Rec/model/hrqvae.py 兼容 task301 ckpt (恢复 log_r/hyp_mean/hyp_scale/encoder)
- 选项 B: 修订 task301_issue30_gate1_stage1_train.py 匹配新 HRQVAE 签名 (移除 curvature_list 等)
- 选项 C: 重启 Stage 1 重新训练 (从零开始, ~90 min)
- 选项 D: 关闭 Issue #34 / 暂缓 task #315

## 4. R11.3 透明

- 这个 NO-GO 不是 D9 多样 hash 机制本身的失败 — 是**上游基础设施变动的副作用**
- task301 SID file (`Instruments_t5_hrqvae_issue30_per_layer_transforms.npy`) 仍然存在
  (Jul-30 00:06:35 生成, 还在磁盘上), 仍可作 Stage 3 输入 (但等于 Issue #30 端点)
- 任何后续 D9 任务需要先解决上游 HRQVAE 不兼容问题
- 没启动 Stage 3 + Stage 4 (避免无意义 GPU 占用)

## 5. 关键产物

- `descriptions/task315_issue34_d9_perlayer_hash_simplified.md` (R11.5 简化决策)
- `scripts/task315_issue34_d9_gate1_stage1_train.sh` (无法运行, 沿用 task301 脚本失败)
- `scripts/task315_issue34_d9_gate2_topk_codebook.py` (Phase 2 加载任务失败)
- `products/task315/hrqvae_issue34_d9_gate1` (symlink → task301/hrqvae_issue30_gate1)
- `verdicts/task315_issue34_d9_nogo.md` (本文件)

## 6. 关联

- Issue #34 (D9 多样 hash, OPEN) — owner 暗示 D9 + 仍未实证
- Issue #30 (r_l=[0.1,1,10]+s_l=[2,2,2] R@10=0.1022 GO marginal) — Stage 1+2 ckpt 基础
- Task #301 (Issue #30 PI) — Stage 1+2 训练链 (旧 HRQVAE 版本)
- Task #307 (Issue #34 前次 wrapper class 尝试) — wrapper class bug Gate 1 FAIL
- HG-Rec/model/hrqvae.py (2026-07-30 00:03 修改) — 上游架构变动

result: Task #315 — Issue #34 D9 multi-hash diversity Gate 1 wrapper class bug NO-GO
