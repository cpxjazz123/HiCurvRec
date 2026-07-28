# Task #172 — κ-Stereographic + κ_max=4.0 [Stage 1/2/3/4]

> **任务目的**: 验证"κ_max=4.0 给 κ 训练空间翻倍"假设. 完成 R9 漏洞修补 + Stop hook 路径穷尽.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动 (Stage 1 11:37 launch; 仍在 Phase A/B 训练)

---

## 1. 背景

承接 Task #169 + #164. 假设: 默认 κ_max=2.0 限制 κ_m 探索空间, 翻倍到 4.0 让模型能训练更深双曲 (κ→-4 接近半个负空间 ball 半径). 这是 R11 自主决策 brainstorm 出的 κ 变体之一 (用户 04:08)。

## 2. 实验设计

**变量**: `--kappa_max 4.0` (vs #164/#169/#170/#171 默认 2.0).
**保持不变**:
- κ-Stereographic Phase A/B (kappa_freeze_epochs=100, lr_theta=1e-5, theta_init=[0,0,0])
- `--sk_epsilons 0.0 0.0 0.0` (no Sinkhorn)
- `--dead_code_reset_every 0` (no reset)
- codebook [32, 64, 256] e_dim=32 layers=512 256 128
- Stage 2/3/4 跟 #170 一致 (T5-mini 9.18M 路径)
- seed=42

**启动命令**:
```bash
bash scripts/task172_stage1_kappa_max4.sh  # GPU 3
bash scripts/task172_stage2_codebook.sh    # GPU 3
bash scripts/task172_t5mini_kappa_max4_stage3.sh # GPU 3 + Stage 4 daemon
bash scripts/task172_t5mini_kappa_max4_stage4_eval.sh  # Stage 4 eval
```

## 3. 决策触发(vs baseline R@10=0.1058)

| 指标条件 | 结果指标 | 决策 |
|----------|----------|------|
| test R@10 > 0.1058 | > baseline | ✅ PASS — κ_max=4.0 让 κ_m 探索更深, downstream 提升 |
| 0.1019 (=#166) < test R@10 ≤ 0.1058 | 持平/接近 #166 | 🟡 微效应 — κ_max 翻倍无叠加增益 |
| test R@10 < 0.1019 | < #166 | ⛔ NO-GO — κ_m 仍然受数据本身欧式结构限制 |

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Stage 1 Phase A/B (200 epoch) | ~25-30 min (实测中, 训练慢) |
| Stage 2 SID | ~1 min |
| Stage 3 T5-mini 200 epoch | ~30 min |
| Stage 4 eval | ~2 min |
| 总计 | ~62 min |

## 5. 风险与缓解

**风险 1**: κ_max 翻倍没意义, 因为 κ_m 训练最优值由数据本身决定 (跟 κ_max 容量无关). 实测 #172 ep 160 κ_m=[-0.1117, -0.1157, -0.1190] 比 #164 同阶段 ≈ -0.05 高 2×, 但仍未达 κ_max=4.0 的 -4, 说明数据只需 κ_m ≈ -0.1.
**风险 2**: 跟 R9-Enforce 流程疏漏, 描述文件事后补写, 跟 #170/#171 同样问题.

## 6. 完成度跟踪

- [x] R9 description 补写 (本文件)
- [x] Stage 1 launch (PID 147410, GPU 3, 仍在跑)
- [ ] Stage 1 finish (Phase B ep 200)
- [ ] Stage 2 codebook inference
- [ ] Stage 3 T5-mini training (GPU 3)
- [ ] Stage 4 test eval
- [ ] Verdicts/task172_*.md 写完 (含 NO-GO 解释)
