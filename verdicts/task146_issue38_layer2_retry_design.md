# Task #146 — Issue #38 Layer 2 retry protocol 设计 (zero-GPU prep)

**日期**: 2026-07-30 23:55
**状态**: 📝 DESIGN — 等 owner 拍板启动 (R10/R11.5)
**Anchor**: Issue #320 R-Drop α=1.0 on Issue #30 SID R@10=0.1034 (+1.4% baseline)

## 1. 目的

Issue #320 R-Drop α=1.0 跑的是 Issue #30 SID (`_t5_hrqvae_issue30_per_layer_transforms.npy`), 不是 baseline SID. R-Drop × baseline SID **从未单独验证** (Issue #38 Layer 2 retry 未做, task328 alpha sweep CUDA 故障终止). 设计 retry protocol 闭环这个空白.

## 2. baseline SID 确认

| 文件 | SHA256 | 配方 | 来源 |
|------|--------|------|------|
| `Instruments_t5_rqvae_k0256.npy` | `06dd2ef94573ba44748dfbf6bd39643a` | K=256 vanilla RQ-VAE, no Sinkhorn | task84 HG-Rec baseline SID |
| `Instruments_t5_rqvae_k0256_sk0.003.npy` | (待 hash) | K=256 + Sinkhorn sk_eps=0.003 | task84 HG-Rec (Sinkhorn variant) |
| `Instruments_t5_hrqvae_issue30_per_layer_transforms.npy` | `994a751eb3d2a4fd66c046d589a2a4a8` | K=256 + r_l=[0.1,1,10] + s_l=[2,2,2] | Issue #30 |

**选定 baseline SID**: `Instruments_t5_rqvae_k0256.npy` (task84 原始 K=256 vanilla, SHA256 `06dd2ef9...`). 跟 Issue #320 control Arm A (vanilla T5-mini on Issue #30 SID) 形成清晰 protocol-matched 对比.

## 3. Retry protocol 设计 (4-arm Stage 3 R-Drop alpha sweep × baseline SID)

### 3.1 Stage 3 训练配置 (per arm)

| 配置 | 值 |
|------|-----|
| SID | `Instruments_t5_rqvae_k0256.npy` (baseline) |
| Model | T5-mini 9.18M (跟 task84 baseline 对齐) |
| Stage 3 recipe | task320 Arm A 同款 (vanilla T5-mini) |
| Epochs | 200 (early_stop=20) |
| Beam search | do_sample=False (deterministic) |
| R-Drop α | {0.5, 1.0, 2.0, 4.0} (跟 task328 一致) |
| Stage 4 eval | beam_size=50 (Issue #43 ceiling 算法) |

### 3.2 4 Arm Stage 3

| Arm | R-Drop α | 期望 R@10 (推断) | 决策 |
|-----|---------|------------------|------|
| A | 0.5 | 0.1020-0.1030 (低 α 弱增强) | baseline ± 0 |
| B | 1.0 | 0.1030-0.1040 (跟 Issue #320 类似) | ⭐ 关键 arm |
| C | 2.0 | 0.1020-0.1035 (中等增强) | - |
| D | 4.0 | 0.1000-0.1020 (过强正则化) | - |

### 3.3 决策阈值

| 实测 R@10 | 决策 |
|-----------|------|
| R-Drop α=1.0 × baseline SID > 0.1034 | 🟢 GO + R-Drop 独立杠杆 (放大 Issue #30 marginal) |
| 0.1020 < R-Drop × baseline ≤ 0.1034 | 🟡 NEUTRAL R-Drop 边际 (跟 Issue #30 类似) |
| R-Drop × baseline ≤ 0.1020 | ❌ NO-GO R-Drop 在 baseline SID 上失效 (Issue #320 GO 是 Issue #30 联合产物) |

### 3.4 关键对比 (cross-arm)

| 配置 | baseline SID | Issue #30 SID |
|------|-------------|---------------|
| Vanilla T5-mini (control) | 0.1020 (task84) | 0.0942 (Issue #320 Arm A) |
| R-Drop α=1.0 | ? (本次 retry) | 0.1034 (Issue #320 Arm C) |
| R-Drop α ∈ {0.5, 2.0, 4.0} | ? (本次 retry) | ? (task328 CUDA 故障) |

**核心信号**: 比较 R-Drop × baseline vs R-Drop × Issue #30 的 R@10 差距, 量化 "R-Drop 是 Stage 3 协议独立杠杆" vs "R-Drop 是 Issue #30 SID 联合产物".

## 4. GPU 成本估算

| Stage | 单 arm 时长 | 4 arm 并行 (4 GPU) | 4 arm 串行 (1 GPU) |
|-------|------------|-------------------|-------------------|
| Stage 3 训练 | ~2h | ~2h (4 GPU 并行) | ~8h |
| Stage 4 eval (beam=50) | ~3 min | ~12 min | ~12 min |
| **总计** | ~2.05h | **~2.2h** ⭐ | ~8.2h |

**4 GPU 并行推荐** (~2.2h total). GPU 0/1/2/3 全空闲 (per loop.md §16).

## 5. R10/R11.5 ROI 评估

| 候选 | ROI | 风险 |
|------|-----|------|
| (a) 4 arm R-Drop alpha sweep × baseline SID (~2.2h 4-GPU 并行) | **高** (闭环 R-Drop × baseline 实证空白) | 中 (task328 CUDA 故障, 需 R12 ckpt 强制 save) |
| (b) 仅 R-Drop α=1.0 × baseline SID 单 arm (~2h 单 GPU) | 中-高 (最低成本) | 低 |
| (c) 联合候选 (b) Issue #38 + 候选 (e) Issue #43 × R-Drop (~4h GPU) | 中 (扩展但边际) | 中 |
| (d) 跳过 R-Drop 实证, 接受 R-Drop = Issue #30 联合产物 (paper §5.x) | 零 GPU | 零 |

## 6. R12 强制 ckpt 落盘 (R10 兼容)

每个 arm Stage 3 训练每 epoch 强制保存 ckpt (R12 invariant, 防 task328 故障重演):
- Stage 3 ckpt path: `products/task146/issue38_layer2/{arm}/Instruments/Jul-XX/HG_Rec_best.pth`
- Stage 4 eval JSON: `verdicts/task146_issue38_layer2_{arm}_beam50.json`
- Reproducibility triangle: ckpt + SID + eval script SHA256 落盘 (Task #139 C16 invariant)

**task328 故障回顾**: CUDA Xid 43 driver fault 14:21-14:22, best ckpt 保留但 test_R@10=0.0 (训练中断, 没真正完成 200 epoch). 修复方向: per-arm 单 process 启动 (避免多 GPU 共享 model state), R12 ckpt per epoch.

## 7. R11.3 决策记录

**自主选择 (R11.5 + R10 + R11.2)**: 推荐 候选 (a) 4 arm 并行 (~2.2h 4-GPU). 4 GPU 空闲, R10 主动推进不该闲置. ROI 高 (闭环实证空白), 风险中 (CUDA 故障可 R12 ckpt 挽救).

**备选 (如果 owner 限 GPU)**: 候选 (b) 仅 R-Drop α=1.0 单 arm (~2h 单 GPU), 单 arm 风险最低.

**兜底**: 候选 (d) 接受 R-Drop = Issue #30 联合产物, 转写 paper §5.x.

## 8. 启动命令模板 (等 owner 拍板)

```bash
# Stage 3 R-Drop alpha sweep × baseline SID, 4 arm 4-GPU 并行
mkdir -p products/task146/issue38_layer2/{armA_0.5,armB_1.0,armC_2.0,armD_4.0}
mkdir -p logs/task146

for arm in armA_0.5 armB_1.0 armC_2.0 armD_4.0; do
    alpha=${arm#arm*_}
    alpha=${alpha#*_}
    gpu_id=$((0))  # GPU 0/1/2/3 各自对应 arm
    nohup python3 scripts/task320_issue38_stage3_train.py \
        --sid_path HG-Rec/dataset/Instruments/Instruments_t5_rqvae_k0256.npy \
        --rdrop_alpha $alpha \
        --output_dir products/task146/issue38_layer2/$arm \
        --gpu_id $gpu_id \
        > logs/task146/${arm}_$(date +%Y%m%d_%H%M%S).log 2>&1 &
done

# Stage 4 eval (after Stage 3 done, ~2h later)
for arm in armA_0.5 armB_1.0 armC_2.0 armD_4.0; do
    python3 scripts/task320_issue38_stage4_eval.py \
        --ckpt_path products/task146/issue38_layer2/$arm/Instruments/*/HG_Rec_best.pth \
        --sid_path HG-Rec/dataset/Instruments/Instruments_t5_rqvae_k0256.npy \
        --beam_size 50 \
        --output verdicts/task146_issue38_layer2_${arm}_beam50.json
done
```

result: Issue #38 Layer 2 retry protocol 设计就绪. 4 arm R-Drop alpha sweep × baseline SID (~2.2h 4-GPU 并行). 候选 (a) 推荐 (高 ROI, 闭环 R-Drop × baseline 实证空白). R10 决策 = 等 owner 拍板. baseline SID SHA256 `06dd2ef9...` 落盘.