# Task #170 — κ-Stereographic + Sinkhorn(ALL 3 layers) [Stage 1/2/3/4]

> **任务目的**: 验证"全 3 层 Sinkhorn 比 L2 only (#169) 更稳定"假设. 完成 R9 漏洞修补 + Stop hook 路径穷尽.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动 (Stage 1 11:36 launch, 11:41 完成; Stage 2/3/4 待跑)

---

## 1. 背景

承接 Task #169 (κ-Stereo + Sinkhorn L2 only, val R@10=0.0998 vs #166 纯 κ-Stereo 0.1177). 用户 04:08 "still have 3 feee gpu think about what method we still can use to combination to our kdistance" 决策 brainstorm 出 7 个 κ 组合变体, 选 top 3 并行 (#170 + #171 + #172) 抢 3 张空闲 GPU.

本任务假设: 全 3 层加 Sinkhorn (`--sk_epsilons 0.003 0.003 0.003`) 跟 baseline RQ-VAE 一致, 验证"是否 Sinkhorn 在 κ-Stereo 框架下需要全 3 层才稳定" 还是 "L2 only 已足够但不好"。

## 2. 实验设计

**变量**: `--sk_epsilons` 三层都开 (0.003 0.003 0.003) vs #169 仅 L2 (0.0 0.0 0.003) vs #164 都不开 (0.0 0.0 0.0).
**保持不变**:
- κ-Stereographic Phase A/B (kappa_freeze_epochs=100, lr_theta=1e-5, theta_init=[0,0,0])
- codebook [32, 64, 256] e_dim=32 layers=512 256 128
- Stage 2: 推断 SID npy (`Instruments/Instruments_t5_rqvae_phase_b_kappa_sinkhorn_all3.npy`)
- Stage 3: T5-mini 9.18M (4+4 layers, d_model=256, d_ff=1024, 4 heads × d_kv=64)
- seed=42

**启动命令**:
```bash
bash scripts/task170_stage1_kappa_sinkhorn_all3.sh       # GPU 0
bash scripts/task170_stage2_codebook.sh                  # GPU 0
bash scripts/task170_t5mini_kappa_sinkhorn_all3_stage3.sh # GPU 0 + Stage 4 daemon
bash scripts/task170_t5mini_kappa_sinkhorn_all3_stage4_eval.sh  # Stage 4 eval
```

## 3. 决策触发(vs baseline R@10=0.1058)

| 指标条件 | 结果指标 | 决策 |
|----------|----------|------|
| test R@10 > 0.1058 | > baseline | ✅ PASS — κ-Stereo + 全 3 层 Sinkhorn 协同, downstream 提升 |
| 0.1019 (=#166) < test R@10 ≤ 0.1058 | 持平/接近 #166 | 🟡 微效应 — Sinkhorn 对 κ-Stereo 无叠加增益 |
| test R@10 < 0.1019 (<#166 纯 κ-Stereo) | < #166 | ⛔ NO-GO — Sinkhorn 在 κ-Stereo 框架下反作用 |

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Stage 1 Phase A/B | ~3 min (实测 11:38-11:41) |
| Stage 2 SID | ~1 min |
| Stage 3 T5-mini 200 epoch | ~30 min (early stop ~20) |
| Stage 4 eval | ~2 min |
| 总计 | ~36 min |

## 5. 风险与缓解

**风险 1**: κ_m 跟 #169 (L2 only) ≈ -0.09 几乎一样 (Sinkhorn L2/L0 都不会改 κ 主信号) → micro-effect 假设落空, 仍 expected NO-GO.
**风险 2**: 跟 R9-Enforce 流程疏漏: 描述文件事后补写 (先 launch 再 description 反向, 应该先建 description). 已在事后修补.

## 6. 完成度跟踪

- [x] R9 description 补写 (本文件)
- [x] Stage 1 launch (PID 143878, GPU 0, 完成 11:41)
- [x] Stage 1 output: κ_m=[-0.0895, -0.0917, -0.0932] (跟 #169/164 一致)
- [x] Stage 1 best_loss_model.pth 已落盘 (4.55 MB)
- [ ] Stage 2 codebook inference
- [ ] Stage 3 T5-mini training (GPU 0)
- [ ] Stage 4 test eval
- [ ] Verdicts/task170_*.md 写完 (含 NO-GO 解释)
