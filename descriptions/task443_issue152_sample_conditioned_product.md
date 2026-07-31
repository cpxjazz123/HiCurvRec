# Task #443 / Issue #152 [方向B Gate1] 样本条件product权重与容量硬分配审计

## R18 4 维度路径对比 (vs Issue #149 / Task #439 + #144 / #146)

| 维度 | Issue #149 (task439 PARTIAL) | Issue #144 (task434 NO-GO) | Issue #146 (task436 PARTIAL) | Issue #152 (本 task, 修复方向B) |
|------|-------------------------------|----------------------------|------------------------------|------------------------------------|
| **D1 spec 摘录** | continuous soft-anchor product, T 温度预注册, F.softplus margin, 但 usage<=3.13%, max_load≈100% | temperature simplex mixing, mixing grad=0 | product distance contrastive, argmin 切断 + relu 0 grad | **sample-conditioned product weights + 容量硬分配**: 每个样本以其三层连续几何特征生成 受熵下界约束的 product weights, mixing 通过连续 product calibration loss 获梯度, 容量器决定硬 SID (Hungarian/auction with cap) |
| **D2 实施核心** | continuous soft-anchor (NO argmin) + softplus margin + random neg | 温度受控 + 熵下界 + softmax mixing | argmin + relu margin + softmax mixing | **SampleConditionedProductKappaModel**: 输入 encoder z_e + 三层 κ → per-sample MLP → 3 weight logits (受 entropy regularization lower bound) → softmax weights (per-sample α_l ∈ Δ). product d = α_0·d_hyp_l0 + α_1·d_hyp_l1 + α_2·d_eucl. hard SID via capacity-constrained Hungarian assignment (跟 #151 同) |
| **D3 Gate 1 失败机制** | 两分量贡献不足 + usage 不达标 (跟 #148 同 collapse family) | mixing grad=0 | argmin 切断 + relu 0 grad | **新机制**: per-sample weights 让 α 随 sample 变化, 不再静态 scalar. 硬容量分配消除单码字捷径. 预期: 三层 usage>=90%, max_load<5%, sample weights 非退化 |
| **D4 引用文献** | arXiv:2307.04514 (同 #146, weighted mixed-curvature) | arXiv:2307.04514 + Riemannian | arXiv:2307.04514 | arXiv:2307.04514 (同 #149/#146, 不同实施) |

**R18 判定**: 4 维度都有差异 (重点在 D1 sample-conditioned weights + 容量硬分配 vs continuous soft-anchor + D2 per-sample MLP weights vs static scalar + D3 反 #149 collapse family + D4 同文献不同实施), 必须做新实验, **不允许**套用 #149/#144/#146 判决 (R18 强制).

## 实施
- `scripts/task443_issue152_sample_conditioned_product.py` (~750 lines, fork from task439 + task442)
- SampleConditionedProductKappaModel: encoder z_e (B, D) → per-sample MLP → 3 weight logits → softmax (受 entropy reg) → per-sample product weights
- product distance: d_mix = α_0·d_hyp_l0 + α_1·d_hyp_l1 + α_2·d_eucl (per-sample α)
- continuous product calibration loss: soft-anchor + softplus margin (跟 #149 同, 但用 sample-conditional α)
- hard SID via Hungarian capacity assignment (跟 #151 同)
- autograd graph proof: aux_loss → κ/mixing grad path + hard assignment branch isolated
- 1000-step static-mixing control (跟 #144/#146 复现) + 1000-step with sample-conditioned + capacity
- 每 100 步记录: sample weights entropy/variance / 三分量贡献 / κ/mixing grad/delta / 容量可行率 / usage / max_load / assignment-cost gap / #47 residual / NaN-Inf
- 8 件套: config + SHA256(item_emb) + d_mix_graph_proof + train_curve + precheck + ckpt + verdict.json + commit

## Precheck 决策阈值 (Issue #152 spec)
- 图: continuous calibration → sample weights/κ, 整数 capacity assignment → hard SID
- 容量求解整型性: assignment 是 int tensor, 每码字 count ≤ cap
- 容量可行性: 100% feasible (所有 batch)
- aux_graph proof: continuous aux → κ/mixing grad path 非零
- hard SID branch isolated

## Gate 1 决策阈值 (Issue #152 spec)
- PASS: 每批容量可行, 三层 usage>=90%, max_load<5%, 每层至少两分量平均贡献>0.1 且 sample weights 非退化, κ/mixing 均有限非零并更新, hard SID round-trip, 无 NaN/Inf, 8 件套齐全
- FAIL: 任一不满足即 STOP

## Gate 4 决策阈值 (Gate 1 PASS 后)
- 六项指标 (R@5/10/20, NDCG@5/10/20)
- 仅 test R@10 > 0.1020 = Target reached

## 复现命令
```bash
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task443 \
  python3 scripts/task443_issue152_sample_conditioned_product.py
```