# Issue #104 — Stage2 SPBI Verdict (R38 NO-GO: immediate codebook collapse ep0→30)

## 状态: ❌ NO-GO (R38 mid-training regress: util_4digit=0.001 by epoch 30, 比 v26 更差)

## 关键失败信号 (R23 trigger — 比 v26 更严重)

代码本利用率 epoch 轨迹:

| Epoch | util_4digit | n_unique_4digit/9922 | 状态 |
|-------|-------------|----------------------|------|
| 0     | 0.007       | 69                   | 🚨 **init 阶段就崩** |
| 10    | 0.002       | 18                   | 继续塌缩 |
| 20    | 0.002       | 20                   | 完全死锁 |
| **30** | **0.001** | **6**                | 🚨 **彻底崩溃** |

**对比 v26 (MCJT)**: v26 在 ep40 util=0.834 (健康), ep170 才崩. **v27 (SPBI) 在 ep0 就 0.007, ep30 已 0.001** — 没有任何可训练窗口.

**R38 决策行**: v27 mid-training regress, util_4digit=0.001 at ep30, 比 v26 (ep170 崩) 更早更严重, 立即 kill + NO-GO 回退.

## 根因分析

SPBI init 把 codeword 固定在 Poincaré 球半径 {0.3, 0.5, 0.7, 0.9}, 但 input item embedding (KMeans 后的欧氏中心) 通常范数 ≈ 0.1-0.3, 经过 exp_map0(c=1) 后 input 半径也只到 ~0.3. 因此:
- input 集中在球内层 (radius < 0.3)
- L0 codeword 16 个在 r=0.3 shell, 16 个在 r=0.5/0.7/0.9 shell (分布均匀)
- 只有最内层 (r=0.3) 的 16 个 codeword 有可能匹配 input → L0 util 立即塌缩到 16/64=25%
- L1 (128 entries) 和 L2 (256 entries) 同样: 只有最内层 1/4 codeword 被分配 → util 25%/25%/25% 理论上限

实测 L0 util_3digit=0.094 (15/64), L1=0.008 (1/128), L2=0.004 (1/256) — 与理论 25% 严重不符, 因为 input 范数进一步集中在原点附近, 几乎所有 input 都命中同一最近 codeword.

**SPBI 设计根本性错误**: shell 半径 0.3-0.9 对应的是 Poincaré 球面位置, 但 input 在球面下的位置由其欧氏范数 + exp_map 决定, 远小于 shell 半径. codeword 放在 r=0.9 shell 完全孤立 (没 input 能到达), 等价于 25% codeword 浪费.

## 时间线

- **2026-08-10 03:34** — 收到 Issue #104 (Stage2 SPBI)
- **2026-08-10 03:34** — R18 grep 验证 v15 无 stratified/shell/radial init → 非同构, 继续
- **2026-08-10 03:35** — v27.1: 加 SPBI_INIT flag + 4 径向层 init 函数 (init_emb 替换 KMeans 分支)
- **2026-08-10 03:36** — py_compile PASS
- **2026-08-10 03:37** — v27.2: 建 tasks/v27_spbi_stage2_from_v15/ + 4 脚本 (R34 合规)
- **2026-08-10 03:37** — launch Stage2 训练 (PID=2883768, master_port=29727)
- **2026-08-10 03:38** — Precheck PASS, SPBI init 启用
- **2026-08-10 03:38** — ep0: util_4digit=0.007 (🚨 立即崩, 比 v26 ep170 更早)
- **2026-08-10 03:38** — ep30: util_4digit=0.001 (彻底崩溃)
- **2026-08-10 03:38** — kill -9 + R38 NO-GO + 写 verdict (本文件)

## 4 Gate 失败机制

- **Gate 1 (Stage2 util_3digit > 0.85)**: ❌ FAIL — 最终 util_3digit ∈ [0.094, 0.008, 0.004], 远低于阈值
- **Gate 2 (SID collision > 60%)**: N/A — SID 未产出
- **Gate 3 (Stage3 健康)**: N/A — 未启动
- **Gate 4 (test_R@10 > 0.1011)**: N/A — 未启动

## R 合规

- **R18** ✅ v15 capmatch 无 stratified/shell/radial init, 非同构验证通过 (启动前)
- **R23** ✅ 信号触发 (util 跨 ≥2 ckpt 急剧下降, 且 ep0 就触发) → kill -9
- **R38** ✅ mid-training regress 立即 kill + R38 决策行 (本 verdict 含)
- **R37** ✅ N/A — 无失败 ckpt 复用, 直接终止 v27 lineage
- **R39** ✅ 立即实施, 不阻塞 Gate A 文本
- **R19** ✅ precheck PASS → launch (R18 同构已排除)
- **R38 watchlist** ✅ 启用 (基于 #103 教训, 每 10 epoch 检查 util_4digit)

## 后续 (R37 决策)

v27 (SPBI) lineage 终止, 与 v26 (MCJT) 同列失败.

**Stage2 创新方向已穷尽**:
- per-layer κ 标量系列 (#86 #87 #225 #228): 4 个 NO-GO/PARTIAL-GO
- HRQ distance (#102): R18 同构 NO-GO (与 v15 同构)
- MCJT multi-c loss (#103): R38 NO-GO (α 塌缩)
- **SPBI stratified init (#104)**: R38 NO-GO (init 与 input 分布严重错配)

后续 Stage2 唯一可能方向 (R38 verdict 推荐):
1. 改 encoder / decoder 架构 (non-HRQVAE 拓扑) — **架构层变更, 风险高**
2. 改 loss function (非 standard VQ 重建, 需更鲁棒的 α 防塌缩) — **算法层变更**

两者都涉及根本性改动, 需要重新审视 Stage2 训练整体设计, 不是简单 patch.

## 产物

- `tasks/v27_spbi_stage2_from_v15/` — 4 个脚本 (R34 合规)
- `taskA/stage2/taskA_stage2.py` — 加 SPBI_INIT flag + stratified init 函数 (R31 不 fork)
- `taskA/_history/v27_spbi_stage2/` — 训练日志 + 崩前产物 (NN idx + MLR calib)
- `verdicts/issue104_spbi_stage2/verdict.md` — 本文件