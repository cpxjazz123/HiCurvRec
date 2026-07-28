# Task #171 — κ-Stereographic + dead_code_reset_every=10 [Stage 1/2/3/4]

> **任务目的**: 验证"定期 reset 死码跟 κ 几何兼容"假设. 完成 R9 漏洞修补 + Stop hook 路径穷尽.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动 (Stage 1 11:36 launch, 11:40 完成; Stage 2/3/4 待跑)

---

## 1. 背景

承接 Task #169 (κ-Stereo + Sinkhorn L2 only) + Task #164 (纯 κ-Stereo baseline). 假设: κ-Stereo + dead_code_reset_every=10 协同, 每 10 epoch 重置 kmeans_init 死码, 让 codebook 健康 + κ 几何兼容, downstream 提升. 这是 R11 自主决策 brainstorm 出的 κ 变体之一 (用户 04:08)。

## 2. 实验设计

**变量**: 加 `--dead_code_reset_every 10` (vs #164/#169/#170 的 0 = 默认不 reset). 每 10 epoch 自动重置利用率最低的码字 (kmeans_init 在 latent space).
**保持不变**:
- κ-Stereographic Phase A/B (kappa_freeze_epochs=100, lr_theta=1e-5, theta_init=[0,0,0])
- `--sk_epsilons 0.0 0.0 0.0` (no Sinkhorn)
- `--kappa_max 2.0`
- codebook [32, 64, 256] e_dim=32 layers=512 256 128
- Stage 2/3/4 跟 #170 一致 (T5-mini 9.18M 路径)
- seed=42

**启动命令**:
```bash
bash scripts/task171_stage1_kappa_dead_code_reset.sh  # GPU 2
bash scripts/task171_stage2_codebook.sh               # GPU 2
bash scripts/task171_t5mini_kappa_dead_code_stage3.sh # GPU 2 + Stage 4 daemon
bash scripts/task171_t5mini_kappa_dead_code_stage4_eval.sh  # Stage 4 eval
```

## 3. 决策触发(vs baseline R@10=0.1058)

| 指标条件 | 结果指标 | 决策 |
|----------|----------|------|
| test R@10 > 0.1058 | > baseline | ✅ PASS — dead_code reset 跟 κ 几何协同提升 |
| 0.1019 (=#166) < test R@10 ≤ 0.1058 | 持平/接近 #166 | 🟡 微效应 — reset 对 κ 无叠加增益 |
| test R@10 < 0.1019 | < #166 纯 κ-Stereo | ⛔ NO-GO — reset 反而扰动 κ |

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Stage 1 Phase A/B | ~3 min (实测 11:38-11:40) |
| Stage 2 SID | ~1 min |
| Stage 3 T5-mini 200 epoch | ~30 min |
| Stage 4 eval | ~2 min |
| 总计 | ~36 min |

## 5. 风险与缓解

**风险 1**: dead_code_reset 可能在 Stage 1 后期已经让 κ_m 滑到 0 (因为 dead code 在 reset 后分布均匀, 强制数据在欧式均衡), 训练到 Phase B 反而把 κ 拉回 0. 实测 κ_m=[-0.0884, -0.0907, -0.0928] 跟 #164 一致, 所以 reset 没改变 κ 主信号, micro-effect.
**风险 2**: 跟 R9-Enforce 流程疏漏, 描述文件事后补写, 跟 #170 同样问题.

## 6. 完成度跟踪

- [x] R9 description 补写 (本文件)
- [x] Stage 1 launch (PID 143879, GPU 2, 完成 11:40)
- [x] Stage 1 output: κ_m=[-0.0884, -0.0907, -0.0928] (跟 #164/#169 一致)
- [x] Stage 1 best_loss_model.pth 已落盘 (4.55 MB)
- [ ] Stage 2 codebook inference
- [ ] Stage 3 T5-mini training (GPU 2)
- [ ] Stage 4 test eval
- [ ] Verdicts/task171_*.md 写完 (含 NO-GO 解释)
