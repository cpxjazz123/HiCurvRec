# Task #175 — κ-Stereographic + κ LOCKED at Ollivier ORC 实测值

> **任务目的**: 验证假设"如果 κ 直接锁定在 Toys 数据真实 ORC 测出来的 hyperbolic 值上 (而非学出来), 下游 Recall 是否能超 Euclidean baseline 0.1058"
> **承接**: Task #70 Ollivier Ricci Curvature 实测 (G0=-0.653, G1=-0.829, G2=-0.840, G3=+0.196, G4=-0.667), Task #164/#169/#170/#171/#172/#174 8 个 κ-Stereo 变体全部 NO-GO (学出来 κ_m ≈ 0), Task #71 init=[-1,0,+1] ablation NO-GO
> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

**用户 hypothesis (2026-07-25)**: "如果直接把 κ 锁定在真实 ORC 测出来的值 (-0.65 ~ -0.84) 上, 不学习, 跑一次, 能不能超过 baseline. 试一下"

**核心 rationale**:
- Task #164-#174 一系列实验发现: 让 κ_m 学, 模型总是把 κ_m 学到 ≈ 0 (Euclidean), Toys 数据上本质欧式最优
- Task #71 init κ ablation 同样: init=[0,0,0] > init=[-1,+1]
- 但 Task #70 ORC 实测表明 Toys 数据**本质双曲** (G1 interaction κ=-0.829, 100% 边 ORC < 0)
- Hypothesis: 量化器在 L2 隐空间学不到曲率, 但**强制**它用真实 ORC 几何, codebook 应该在 hyperbolic 空间聚类, 下游可能更好

**区别于之前实验**:
- Task #164-#174: θ_m learnable → kappa_m 学到 ≈ 0 (数据驱动选欧式)
- **Task #175**: θ_m **FROZEN**, kappa_m 永远 = 真实 ORC, **绕过"数据驱动选 κ→0"陷阱**

---

## 2. 实验设计

**变量**: κ_m 锁定方式 (本任务 = LOCKED at ORC, 不学)
**保持不变**:
- Stage 1 RQ-VAE 架构 (encoder/decoder, FreeCurvHRQVAE, num_hierarchies=3)
- Stage 1 训练流程 (200 epoch, batch_size, lr)
- Stage 2 SID codebook inference (rkmeans_inference_flat, Sinkhorn L2 only)
- Stage 3 T5-mini 9.18M 训练 (200 epoch + early stop)
- Stage 4 test eval (beam_size=20, topk=[5,10,20])
- 数据集 (Toys)
- Seed (42)

**κ 配置 (per-layer per-component 锁定)**:

| Layer | Component 0 κ | Component 1 κ | Component 2 κ | 来源 |
|-------|--------------|--------------|--------------|------|
| L0 | -0.653 | -0.829 | -0.840 | G0 attr / G1 interact / G2 cooccur |
| L1 | -0.653 | -0.829 | -0.840 | 同上 (per-layer 独立) |
| L2 | -0.653 | -0.829 | -0.840 | 同上 |

**为什么用 M=3**: 用户 design §1 建议 M=3 时 e_dim=32 → 11+11+10 维分段, 3 component 可分配 3 个 ORC 值. **排除 G3 copurchase (+0.196 spherical)** 避免"球面/双曲混搭"噪声, 只用 3 个 hyperbolic ORC 值.

**Stage 1 启动命令**:
```bash
cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec
python3 -u /home/wlia0047/ar57/wenyu/GeneRec/scripts/task175_orc_locked_stage1_train.py \
    --epochs 200 --batch_size 1024 --lr 1e-3 --num_hierarchies 3 \
    --kappa_orc_values -0.653 -0.829 -0.840 \
    --output_dir /home/wlia0047/ar57/wenyu/GeneRec/products/task175/orc_locked/
```

**Stage 2 (codebook inference)**:
```bash
python3 -m src.inference experiment=rkmeans_inference_flat \
    data_dir=/home/wlia0047/ar57/wenyu/GeneRec/data/amazon_data/toys/ \
    output_file=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_orc_locked.npy \
    inference.lengths=[32,64,256,1] inference.batch_size=1024 \
    inference.rqvae_checkpoint=/home/wlia0047/ar57/wenyu/GeneRec/products/task175/orc_locked/<TS>/best_loss_model.pth
```

**Stage 3 (T5-mini 9.18M)**:
```bash
cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec
python3 -u /home/wlia0047/ar57/wenyu/GeneRec/scripts/task175_orc_locked_stage3_train.py \
    --epochs 200 --batch_size 256 --lr 1e-4 \
    --code_path _t5_rqvae_orc_locked.npy \
    --output_dir /home/wlia0047/ar57/wenyu/GeneRec/products/task175/t5mini_orc_locked/
```

**Stage 4 (test eval)**: 复用 #170/#174 working pattern (config dict + GenRecDataset positional args).

---

## 3. 决策触发 (vs baseline 0.1058)

| 指标条件 | 结果指标 | 决策 |
|----------|----------|------|
| test R@10 > 0.1058 | **🟢 GO** | κ LOCKED 验证成功, 下游超 baseline. 写 verdict 报告. |
| 0.1000 ≤ test R@10 ≤ 0.1058 | 🟡 MARGINAL | 比 baseline 略低 (-1% to -5%), 仍 NO-GO, 但跟 #171 -5.4% 同档, 写 verdict 说明 κ LOCKED 接近 baseline. |
| test R@10 < 0.1000 | ⛔ NO-GO | κ LOCKED 比 baseline 差 >5%, 假设证伪: 即使强制 ORC 几何, 下游仍无法突破. 写 verdict. |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Stage 1 RQ-VAE 200 epoch (κ LOCKED) | ~30 min (单 Phase, 跳过 Phase A/B split) |
| Stage 2 codebook inference | ~5 min |
| Stage 3 T5-mini 9.18M 200 epoch | ~50 min (early stop ~ep50-80) |
| Stage 4 test eval | ~1 min |
| **总计** | **~85 min (~1.5 h)** |

---

## 5. 风险与缓解

**风险 1**: κ 锁定为负数大值 (-0.84), 距离公式 `arctan(√|κ|·r / |2-κr²/2|)` 中 `2 - κr²/2` 在 |κ|·r²/4 > 2 时会**变号**, 数值不稳定 → 缓解: 用现有 `hrqvae_free_curv.py` 的 `.abs().clamp_min(1e-6)` 防爆
**风险 2**: θ_m FROZEN 但 Adam optimizer 仍会 update 它 (无 requires_grad=False), 浪费 compute → 缓解: 在 __init__ 后立即 `for vq in self.vq_layers: vq.theta_m.requires_grad = False`
**风险 3**: 之前所有 κ-Stereo 变体下游都 < baseline, LOCKED 仍可能 NO-GO → 这是 hypothesis 检验的核心, NO-GO 写 verdict 闭环
**风险 4**: ORC 中 G3 是 +0.196 球面, Toys 数据上 spherical 是否有效未知. 本任务**排除** G3, 只用 3 个 hyperbolic ORC, 保持单一变量.

---

## 6. 完成度跟踪

- [ ] Stage 1 launch (κ LOCKED at ORC)
- [ ] Stage 1 finish (best_loss_model.pth 落盘)
- [ ] Stage 2 SID codebook inference
- [ ] Stage 3 T5-mini launch
- [ ] Stage 3 finish (HG_Rec_best.pth 落盘)
- [ ] Stage 4 test eval (test R@10)
- [ ] 写 verdict (`verdicts/task175_orc_locked_result.md`)
- [ ] 更新 loop.md §16 + §15.4 (per R8)