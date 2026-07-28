# Task #179 — Idea 1 dual-branch hyperbolic T5 (Phase 2 启动)

> **任务目的**: 量化器层完全不动 (复用纯欧式 RQ-VAE + Sinkhorn 已验证稳定的 recipe, phonism R@10=0.1058, 100% 利用率), 把曲率逻辑挪到 Stage 3 T5 encoder 内部做长/短期双分支 (长期: 双曲 expmap0 → Möbius linear; 短期: 标准欧式 T5 self-attention). 验证 dual-branch hyperbolic 在 T5 内部是否真有几何优势.

> **完成日期**: in progress
> **状态**: 🟡 Phase 2 Stage 1 待启动 (R10 主动推进: 不空闲等待 Phase 1 Stage 3)

---

## 1. 背景

**2026-07-25 用户前置指令**: "Idea 1: 量化器层完全不动 (复用纯欧式 RQ-VAE + Sinkhorn 已验证稳定的 recipe), 把曲率逻辑挪到 Stage 3 T5 encoder 内部做长/短期双分支."

**前置 task 结论**:
- Task #84 (旧 HG-Rec baseline): R@10=0.1020 (建立在 broken 代码上)
- Task #178 (Phase 1, in progress): 修正后 HG-Rec baseline, 论文 Table 6 原配 + utils.py 4 处 bug 修复, Stage 3 训练中
- 9 个 κ-Stereo variants (#165-#175 + #176/#177): 全部 NO-GO, 大概率源自 broken baseline

**假设 R1**: 把双曲逻辑从 quantizer (artifact-prone) 挪到 T5 encoder (math clean) 后, 双分支能给出正向信号 — 长期 hyperbolic branch 处理 history-mean pooled semantics, 短期 T5 self-attention 处理 token-level 关系, 互补而不冲突.
**假设 R2**: 若 R1 失败 (R@10 ≤ 修正 baseline), 进一步证实 "几何在 Instruments 上无作用" 是真结论.

## 2. 实验设计

**变量**: 仅 Stage 3 T5 encoder 内部新增 dual-branch fusion (长分支双曲, 短分支欧式)
**保持不变**:
- Stage 1+2: 纯欧式 RQ-VAE + Sinkhorn (EuclideanHRQVAE 子类, 完全去掉 expmap0/logmap0/proj_to_ball, 用欧式 cdist + 欧式 commitment loss)
- T5-small 5.5M backbone (跟 #178 / Task #84 baseline 严格对齐: 6 enc + 4 dec, d_model=128, num_heads=6, d_kv=64)
- Stage 2 SID tensor shape `(9922, 4)`, num_emb_list=[32,64,256,1] (跟 phonism 一致, 4th digit dedup)
- Stage 4 eval 配置 (Recall@5/10/20, NDCG@5/10/20, beam_size=20, seed=42)
- Item embeddings (`sentence-t5-base` 输出, 9922 × 768)

**Stage 1 启动命令** (纯欧式 RQ-VAE 训练):
```bash
cd /home/wlia0047/ar57/wenyu/GeneRec
bash scripts/task179_pure_euclidean_stage1_train.sh
```

**Stage 2 启动命令** (纯欧式 Sinkhorn codebook inference):
```bash
python3 scripts/task179_pure_euclidean_stage2_codebook.py \
  --ckpt_path /home/wlia0047/ar57/wenyu/GeneRec/products/task179/euclidean_baseline/<TS>/HRQVAE_best.pth \
  --output_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_pure_euclidean.npy \
  --device cuda:1
```

**Stage 3 启动命令** (Dual-branch T5 训练, 计划):
```bash
cd /home/wlia0047/ar57/wenyu/GeneRec
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task179_stage3
nohup python3 scripts/task179_dual_branch_stage3_train.py \
  --dataset_name Instruments \
  --dataset_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/ \
  --code_path _t5_rqvae_pure_euclidean.npy \
  --codebook_size 32 64 256 1 \
  --num_epochs 200 \
  --batch_size 256 \
  --lr 1e-4 \
  --kappa_max 2.0 \
  --device cuda:1 \
  --mode train \
  --save_path /home/wlia0047/ar57/wenyu/GeneRec/products/task179/dual_branch_t5/<TS> \
  --log_path /home/wlia0047/ar57/wenyu/GeneRec/logs/task179 \
  --seed 42 \
  --early_stop 20 \
  --beam_size 20 \
  --infer_size 96 \
  > logs/task179_stage3_train.out 2>&1 &
```

**Stage 4 eval 命令**:
```bash
bash scripts/task179_dual_branch_stage4_eval.sh
```

## 3. 决策触发 (vs Task #178 新 baseline)

| 测得 R@10 | 与新 baseline 比较 | 解读 + 下一步 |
|-----------|-------------------|-------------|
| **> 新 baseline + 0.005** | 显著超过 | Dual-branch 在 Instruments 上**有几何优势**, κ-Stereo 思路挪到 T5 encoder 内有效, 后续可探索 full hyperbolic attention. |
| **±0.005 内** | 中性 | Dual-branch 跟新 baseline 持平, 进一步证实 "几何在 Instruments 上无作用" — 整条 κ-Stereo 路径关闭. |
| **< 新 baseline - 0.005** | 显著恶化 | Dual-branch 引入额外复杂度反而拖低, 这条想法本身不成立. |

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Stage 1 纯欧式 RQ-VAE 训练 (200 ep) | ~15-30 min (GPU 1) |
| Stage 2 纯欧式 codebook 推断 | ~5-10 min (GPU 1) |
| Stage 3 Dual-branch T5 训练 (200 ep + early_stop=20) | ~2-4 h (GPU 1) |
| Stage 4 test eval | ~2-3 min (GPU 1) |
| **总计** | **~3-5 h** |

## 5. 风险与缓解

**风险 1**: 纯欧式 Stage 1 训练不稳定 (无 expmap0 数值保护, Sinkhorn 在 high-dim 易下溢)
**缓解**: 沿用 0.003 epsilon + log-sum-exp stable softmax (跟 task178 Phase 0 修复一致), R12 强制 best_loss 落盘

**风险 2**: Stage 3 dual-branch 融合 gating 坍缩 (类似 Task #174 GCN gating 经验教训)
**缓解**: **不用跨分支 attention 融合**, 用单 sigmoid gate 加 pool, 初始化 gate bias=0 让融合初期中性 (类比 task174 修复)

**风险 3**: Dual-branch 跑完后 Stage 4 eval 时 fused hidden states 跟原 T5 不兼容, 报 shape error
**缓解**: Stage 3 forward 内部用统一的 `fused_hidden` 替换 self.model.encoder 输出, 不破坏 lm_head / decoder 接口

## 6. 完成度跟踪

- [ ] EuclideanHRQVAE 类实现 + py_compile PASS
- [ ] Stage 1 纯欧式 RQ-VAE 训练启动 (GPU 1)
- [ ] Stage 1 finish (best_loss_model.pth 落盘)
- [ ] Stage 2 纯欧式 codebook 推断 (9922 unique SID + 4 层利用率 ≥85%)
- [ ] Dual-branch T5 encoder 代码 + Stage 3 launcher (依赖 #178 R@10 baseline)
- [ ] Stage 3 launch + finish (best ckpt 落盘 + 早停触发 / 200 epoch 完成)
- [ ] Stage 4 launch + finish (eval on test set, JSON 写入)
- [ ] verdict 写完 (`verdicts/task179_dual_branch_t5_result.md`)
- [ ] loop.md §16 cleanup (R8)

## 7. R-RQ-VAE framework 集成

### R7 GPU 隔离
- Phase 1 (#178) 占 GPU 0 全程
- Phase 2 (#179) 占 GPU 1, 完全并行不抢卡
- Phase 3 (#180) 占 GPU 2/3 之一

### R8 §16 cleanup
- #176/#177 stale rows 已清
- #178/#179 完成 → 立即删除活跃行, verdict 保留

### R9 编号连续
- descriptions max = 178 → 新编号 179 ✅ 已 pre-creation check

### R10 主动推进
- Phase 2 Stage 1 立即启动 (R10 强制: 不空闲等待 Phase 1 Stage 3)

### R11 自主决策
- Stage 1 num_emb_list=[32,64,256,1] (跟 phonism 一致, 备选 [64,128,256,1] 跟论文一致 — 不选, 跟 phonism 对齐有现成稳定 recipe)
- Stage 1 β=0.25 (van den Oord 标准), ε=0.003 (Sinkhorn on)
- Stage 3 T5 backbone = T5-small 5.5M (跟 #178 baseline 严格对齐)
- Stage 3 dual-branch fusion 用单 sigmoid gate (备选 cross-attention, 不选 — 控制风险)

### R12 强制存 ckpt
- Stage 1 trainer 默认 best_loss_model.pth (跟 task178 一致)
- Stage 3 launcher 沿用 task84_hgrec_stage3_train.py 模板 (R12 验收过)

### R13 不用 worktree
- 全部修改直接落在共享 checkout

## 8. 关键技术细节

### EuclideanVectorQuantization (vs utils.py 的 HVectorQuantization)

**关键差异**:
- ❌ 删除 `expmap0` / `logmap0` / `proj_to_ball` / `poincare_distance`
- ✅ 欧式距离: `d = torch.cdist(latent, codebook)`
- ✅ 欧式 commitment loss: `F.mse_loss(x_q.detach(), latent)` + `F.mse_loss(x_q, latent.detach())`
- ✅ β 挂载: `loss = codebook_loss + β * commitment_loss` (跟 task178 修正版一致)
- ✅ Sinkhorn 用 utils.py 的 sinkhorn_algorithm (log-sum-exp 稳定版本已 Phase 0 修复)
- ✅ straight-through: `x_q = x + (x_q - x).detach()`

### EuclideanHRQVAE 包装

直接复用 `HG-Rec/model/hrqvae.py` 的 `HRQVAE` 类结构, 但把 `HResidualVectorQuantization` 换成 `EuclideanResidualVectorQuantization`, 重写 `compute_loss` 用欧式 MSE.