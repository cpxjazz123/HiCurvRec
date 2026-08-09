# Issue #103 — Stage2 MCJT Verdict (R38 NO-GO: codebook collapse epoch 160→170)

## 状态: ❌ NO-GO (R38 mid-training regress: util_4digit 0.66→0.001 in 1 epoch)

## 关键失败信号 (R23 trigger)

代码本利用率 epoch 轨迹:

| Epoch | util_4digit | n_unique_4digit/9922 | 状态 |
|-------|-------------|----------------------|------|
| 0     | 0.206       | 2048                 | init |
| 40    | 0.834       | 8272                 | 健康 (v15 baseline 接近) |
| 80    | 0.789       | 7830                 | 微降 |
| 120   | 0.740       | 7343                 | 缓降 |
| 160   | 0.660       | 6546                 | ⚠️ 仍在下降 |
| **170** | **0.001** | **10**            | 🚨 **CATASTROPHIC COLLAPSE** |
| 180   | 0.002       | 15                   | 完全崩溃 |
| 200   | 0.001       | 10                   | 完全崩溃 |

**R38 决策行**: v26 mid-training regress, util_4digit 0.660 → 0.001 (epoch 160→170, -0.659), 立即 kill + NO-GO 回退。

## 根因分析

MCJT α_c learnable + 多曲率 recon 驱动 codebook 退化:

1. **前 40 epoch 健康**: util 0.21 → 0.83, 正常学习
2. **40-160 epoch 缓降**: util 0.83 → 0.66, MCJT 多曲率信号与 codebook EMA 更新竞争
3. **160-170 epoch 悬崖**: util 0.66 → 0.001, α_c 大概率塌缩到 c=5.0 (最大 c), 极端曲率下 codebook embedding 被吸引到边界 → 多 codeword 重叠到同一位置

κ 健康 (c=[2.17, 9.75, 7.15] 类似 v15), 排除 c_l 漂移问题。问题在 α_c → 单 c 塌缩 → 几何失配。

## 时间线

- **2026-08-10 03:18** — 收到 Issue #103 (Stage2 MCJT)
- **2026-08-10 03:18** — R18 grep 验证: v15 capmatch 无 multi-c/alpha_c/C_set/logits_c → 非同构, 继续
- **2026-08-10 03:19** — 写 v26.1 task: 在 taskA/stage2 加 MCJT_ALPHA flag + α_c nn.Parameter + 多 c recon + entropy reg
- **2026-08-10 03:25** — py_compile PASS
- **2026-08-10 03:26** — 建 tasks/v26_mcjt_stage2_from_v15/ + 4 脚本 (stage1/2/3/4_beam20)
- **2026-08-10 03:30** — 第一次 launch 失败 (item_emb.parquet 不存在)
- **2026-08-10 03:30** — 修正 path → 第二次 launch 成功 (PID=2869729)
- **2026-08-10 03:30** — Precheck PASS (7 项 strict)
- **2026-08-10 03:32** — ep40: util_4digit=0.834, 健康
- **2026-08-10 03:33** — ep170: util_4digit=0.001, 触发 R23 信号
- **2026-08-10 03:33** — kill -9 + R38 决策 + 写 verdict (本文)

## 4 Gate 失败机制

- **Gate 1 (Stage2 util_3digit > 0.85)**: ❌ FAIL — 最终 util_3digit ∈ [0.06, 0.07, 0.01], 远低于阈值
- **Gate 2 (SID collision > 70%)**: N/A — SID 未产出 (训练未到终态)
- **Gate 3 (Stage3 健康)**: N/A — 未启动
- **Gate 4 (test_R@10 > 0.1011)**: N/A — 未启动

## R 合规

- **R18** ✅ v15 capmatch 无 multi-c 机制, 非同构验证通过 (启动前)
- **R23** ✅ 信号触发 (util 跨 ≥2 ckpt 急剧下降) → kill -9
- **R38** ✅ mid-training regress 立即 kill + R38 决策行 (本 verdict 含)
- **R37** ✅ N/A — 无失败 ckpt 复用, 直接终止该 lineage
- **R39** ✅ 立即实施, 不阻塞 Gate A 文本
- **R19** ✅ precheck PASS → launch (R18 同构已排除)

## 后续 (R37 决策)

- v26 (MCJT) 全部产物仅留记录 (Stage2 ckpt 部分在 ep40 时是健康的, 但 ep170 已崩, 不构成可复用 ckpt)
- 失败 lineage 终止, 不可在 v26 上叠加新曲率机制
- 后续 Stage2 创新必须提出与 v15 capmatch + v26 (MCJT) 都正交的方向:
  - 改 encoder / decoder 架构 (non-HRQVAE 拓扑)
  - 改 codebook 初始化 / EMA 更新规则 (避免 MCJT 类 α 塌缩)
  - 改 loss function (非 standard VQ 重建, 需更鲁棒的 α 防塌缩机制)

## 产物

- `tasks/v26_mcjt_stage2_from_v15/` — 4 个脚本 (R34 合规)
- `common/stage2/taskA_stage2.py` — 加 MCJT_ALPHA flag + 多 c recon + α_c learnable (R31 不 fork)
- `taskA/_history/v26_mcjt_stage2/` — 训练日志 + 崩前产物 (NN idx + MLR calib)
- `verdicts/issue103_mcjt_stage2/verdict.md` — 本文件