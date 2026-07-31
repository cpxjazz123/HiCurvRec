# Task #437 / Issue #147 [方向C Gate3] 曲率条件化T5注入替代零初始化dual-gate

## R18 4 维度路径对比 (vs Issue #142 / #129)

| 维度 | Issue #142 (task432, Adapter R@10=0.03684 NO-GO) | Issue #129 (旧 dual-gate) | Issue #147 (本 task, 修复方向C) |
|------|--------------------------------------------------|---------------------------|------------------------------------|
| **D1 spec 摘录** | Manifest lineage PASS + control R@10=0.10203 ≈ baseline, Adapter 反作用 (zero-init gate) | 旧 zero-init dual-gate: active-down → ReLU 注入 + 跟 T5 embedding 路径不兼容 | **curvature-conditioned residual injection**: 用每层 SID/κ metadata 生成受限 residual scale, T5 encoder input 表示处做 identity-preserving 注入. gate=0 精确退化为原T5, 非零 gate → 有限非零可审计 grad |
| **D2 实施核心** | AdapterHookedHGRec.generate (R18 修复 #136 routing bug) 但底层仍是 zero-init dual-gate | zero-init gate + active-down→ReLU (硬约束 + ReLU 饱和) | **CurvatureConditionedAdapter**: (SID tokens + κ metadata) → small MLP → bounded scale (sigmoid/clamp) → 加到 T5 encoder input embedding residual. gate 参数化: gate = sigmoid(logit) ∈ [0, 1], gate=0 时 residual 被门控=0, 输出跟原T5 max diff=0 |
| **D3 Gate 1 失败机制** | #129 仅 2 epoch + zero-init + active-down→ReLU 路径不兼容 → adapter 反作用 | zero-init 跟 ReLU 联合让 gradient 早期就 dead | **curvature-conditioned residual + identity-preserving 注入假设**: 用 SID/κ metadata 让 adapter 拿到语义相关信号, 跟 T5 embedding 路径兼容, gate 严格可微 + bounded, 不饱和 |
| **D4 引用文献** | #129 旧 dual-gate 内部 spec | 旧 dual-gate 内部 spec | **arXiv:2309.04082《Curve Your Attention》**: product-stereographic 几何操作注入 Transformer 表示/注意力路径. 跟 #142 完全不同文献 |

**R18 判定**: 4 维度都有差异 (重点在 D1 curvature-conditioned residual vs zero-init dual-gate + D2 identity-preserving 注入 + D3 bounded gate + D4 Curve Your Attention), 必须做新实验, 不允许套用 #142/#129 判决.

## 实施
- `scripts/task437_issue147_curvature_conditioned_residual.py` (~340 lines)
- CurvatureConditionedAdapter: MLP (input = SID token embedding + κ metadata, output = bounded scale)
- Wrapper HG_Rec_with_CurvatureAdapter: T5 shared(input_ids) → encoder_input → adapter(encoder_input, SID_metadata, κ_metadata) → T5 encoder blocks
- Precheck 强制 (Issue #147 spec):
  1. gate=0 时 adapter 输出 = 0, 整个 forward 跟原T5 max diff = 0
  2. 非零 gate 时 conditioner/gate gradients 有限非零
  3. 真实 history-SID 输入可加载 (verify SID token range in [0, K_l))
- Stage 3 短训练 (10 epoch): loss 曲线 + 参数量 + adapter-only ckpt SHA256 + save/load missing=0/unexpected=0 + 两次 forward 一致性
- 不做 Stage 4 (per spec)
- 8 件套: config + SHA256(T5 ckpt + SID) + adapter_init_proof + gradient_proof + train_curve + ckpt_SAVE + ckpt_LOAD + commit
- 产物: `products/task437_issue147_curvature_conditioned_residual/{config,verdict,adapter_init_proof,gradient_proof,ckpt_train.json,adapter.pt}`

## Precheck 决策阈值
- gate=0 → adapter 输出 = 0 (max diff = 0)
- gate=0 → forward output 跟原T5 max diff = 0 (tolerance 1e-6)
- 非零 gate → adapter 参数 grad 有限非零
- 真实 history-SID token range 验证: 全部 in [0, K_l)

## Gate 3 决策阈值 (Issue #147 spec)
- PASS: 所有合同通过 + 训练中几何支路确有更新 + 无 NaN/Inf + 真实 history-SID 输入可加载
- 失败即 STOP, 不做 Stage 4

## 复现命令
```bash
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task437 \
  python3 scripts/task437_issue147_curvature_conditioned_residual.py
```