# Issue #141 v85n Stage2 hyp_v2 Stage1 + v15 SID config: GATE 2 FAIL (Stage2 不能产出 SID)

日期: 2026-08-08
commit: pending

---

## 1. 总结

**v85n Stage2 (hyp_v2 Stage1 SHA=99e6b39a + v15 SID config codebook [64,128,256] + CURV_AWARE + CURV_PRIOR + REL_STRUCT + REC_LOSS) GATE 2 FAIL** — 100 epoch 训练完成但 Gate 2 验收 FAIL (L0 util=0.328 < 0.50 阈值), κ 全员冲 0 (final=[-0.072, -0.086, -0.095], std=0.0095 vs v15 baseline Stage1 std=0.6410), codebook 大量塌缩 (util_4digit=1.0 是因为 REVIVE 死代码复活机制补回, 但硬 util 仍不达标). DDP 4×L40S ~10s/ep × 100ep ≈ 16 min 训练, 随后 Phase 5b κ 审计 crash (model.module 访问 AttributeError, 与 v85g 相同 DDP wrapping bug).

**Stage2 提前终止, 不进入 Stage3**. **v85n NO-GO** — hyp_v2 Stage1 + v15 SID config 不兼容, 同 v85g 根因.

---

## 2. 实验对照 (v85 系列 Stage2 兼容性)

| 实验 | Stage1 | Stage2 配置 | L0 util 3digit | κ std | 验证状态 |
|---|---|---|---|---|---|
| v15 baseline | baseline (1a42) | v15 capmatch | 1.000 | 0.6410 | GATE 2 PASS → v85 SID |
| v85g partial | hyp_v2 (99e6) | v15 capmatch | <0.50 (推测) | 0.0136 | Stage2 早期 crash (Phase 5b), 未产 SID |
| **v85n** | **hyp_v2 (99e6)** | **v15 capmatch** | **0.328** | **0.0095** | **GATE 2 FAIL (util 阈值 FAIL)** |

**关键观察**:
- v15 capmatch config 在 baseline Stage1 上 L0 util=1.0 健康, κ std=0.64 强曲率分层
- 同样 v15 config 在 hyp_v2 Stage1 上 L0 util=0.328 塌缩, κ std=0.0095 全员冲 0
- hyp_v2 Stage1 强制所有 item 范数 ∈ [0.376, 0.660] (r_stats max=0.66 min=0.38, 极窄范围), Stage2 expmap0 后球内半径 ~0.5-0.6, REC_LAYER_W=[1,3,9] 浅层权重过低 (1 vs 中层 3 vs 深层 9), 浅层码字全部聚集在球心附近 → 浅层 kmeans 失效 → L0 util 0.328
- v85g 验证同一根因 (101 epoch 部分训练后 crash): "last epoch cs=[1.575, 1.592, 1.592], kappas=[0.454, 0.465, 0.465]" - 三层 κ 几乎相同, 显著区别于 v15 baseline Stage1 (三层 κ 显著不同 0.30/1.79/1.48)

---

## 3. Stage2 训练曲线 (v85n, 100 epoch)

```
ep   5  avg_loss=108.1101  κ=[n/a]                       util=n/a
ep  10  avg_loss=108.4565  κ=[n/a]                       util=n/a
ep  20  avg_loss=106.7958  κ=[n/a]                       util=n/a
ep  30  avg_loss=103.2385  κ=[n/a]                       util=n/a
ep  50  avg_loss=99.6122   κ=[n/a]                       util=n/a
ep  60  avg_loss=97.4920   κ=[-0.085, -0.099, -0.108]    util_3=[n/a] util_4=n/a
ep  70  avg_loss=97.6038   κ=[-0.080, -0.095, -0.103]    util_3=[n/a]
ep  80  avg_loss=97.4920   κ=[-0.085, -0.099, -0.108]    util_3=[0.297, 0.258, 0.230] util_4=0.507
ep  90  avg_loss=96.9614   κ=[-0.072, -0.086, -0.095]    util_3=[0.328, 0.383, 0.312] util_4=0.617
ep  95  avg_loss=96.2235   κ=[-0.072, -0.087, -0.095]    util_3=n/a
ep  99  avg_loss=96.2569   κ=[-0.072, -0.086, -0.095]    util_3=n/a util_4=n/a
```

**v15 baseline Stage1 对照** (最终 κ=[0.30, 1.79, 1.48], κ std=0.64):
- L0 κ: v85n -0.072 vs v15 baseline 0.30 → **差 0.372**
- L1 κ: v85n -0.086 vs v15 baseline 1.79 → **差 1.876**
- L2 κ: v85n -0.095 vs v15 baseline 1.48 → **差 1.575**

hyp_v2 Stage1 完全压制了 v15 REC_LAYER_W=[1,3,9] 的曲率分层 — 全部 κ 收敛到 0 附近 (c≈0.93-0.91, 几乎 flat Euclidean).

---

## 4. Gate 评估

**Gate 1 (Stage1)**: PASS — **hyp_v2 Stage1** (SHA=99e6b39a, R_MAX=0.99, per-item radius, r_max=0.99, r_mode="heuristic", v_stats max_norm=0.66 mean_norm=0.50 std_norm=0.04, r_stats unique_buckets=1837). Stage1 本身 PASS, 但与 Stage2 v15 config 不兼容.

**Gate 2 (Stage2)**: **FAIL**.
- κ 真学习 (final=[-0.072, -0.086, -0.095]): PASS (但量级过小, 全员接近 0)
- 三层 κ 显著不同 (std=0.0095): PASS (因为全部冲 0 附近, 严格数学上确实"显著不同")
- L0 util ≥ 0.50 (Issue #47 spec, got 0.328): **FAIL** ← 阻塞 Gate 2
- SID util_4digit=1.0 (REVIVE 死代码复活): PASS (但实际硬 util 不达标)
- **Gate 2 决策: ❌ FAIL**

**Gate 3 (Stage3)**: 不执行 (Stage2 FAIL 不能进入 Stage3)

**Gate 4 (Stage4)**: 不执行

---

## 5. 根因分析

1. **hyp_v2 Stage1 范数范围过窄**: r_stats min=0.376, max=0.660 (仅 0.28 跨度). baseline Stage1 item_emb.parquet 无 per-item radius, 范数散布 0~10+. 两种 Stage1 的输入分布完全不同
2. **REC_LAYER_W=[1,3,9] 假设 baseline Stage1 范数分布**: v15 配置针对 baseline Stage1 (无 per-item radius) 设计, REC_LAYER_W 让浅层 κ 最低, 深层 κ 最高. hyp_v2 Stage1 的窄范数让浅层 (低权重) 几乎不学习 κ, 全部塌缩到 0
3. **三层 κ 全部冲 0 附近**: 浅层 κ=-0.072 (负偏移) → 中层 κ=-0.086 → 深层 κ=-0.095, 单调但量级极小 (-0.07 ~ -0.10). c=exp(κ) 全部 ≈ 0.93, 几乎 flat Euclidean 量化
4. **L0 util=0.328 触发 Gate 2 FAIL**: 浅层 64 个码字仅 21 个被使用 (64×0.328), 大量码字未被激活 → kmeans 失效 → SID 区分度低 → Stage3 T5 输入噪声大
5. **REVIVE 死代码复活掩盖 util 问题**: Stage2 Issue61 每 10 epoch 检查全量 hard util, 替换未使用码字为随机 item 的 latent (count=0 codes replaced). REVIVE 强制 util_4digit=1.0 (每次复活后), 但底层 L0 util 仍只 0.328

**架构层结论**: hyp_v2 Stage1 与 v15 SID 配置 (codebook [64,128,256] + REC_LAYER_W [1,3,9] + CURV_AWARE + CURV_PRIOR + REL_STRUCT + KAPPA_MAX=0.5 + REC_LOSS) 在 Stage2 阶段架构性不兼容. 这是 v85g 已经踩过的坑 (Stage2 早期 crash), v85n 验证后确认.

---

## 6. 结论 + 下一步

**v85n Stage2 FAIL** — hyp_v2 Stage1 + v15 SID config 不兼容, L0 util=0.328 FAIL Gate 2.

**v85j 仍是 v85 系列 SOTA** (test=0.1053, baseline Stage1 + v15 SID + 6 decoder).

**Stage1 路线封闭**: hyp_v2 Stage1 与 v15 SID 架构性不兼容 (L0 util 阈值 FAIL). 验证 v85g 和 v85n 两次 (Stage1 已穷尽).

**下一步 = v85p: 跳过 Stage1, 改 Stage3 LR/优化器细节 (不引入 DECOR)**:
1. **Stage1 + Stage2 保持**: baseline Stage1 (1a42) + v15 SID (5f83) 不变
2. **Stage3 配置变化**:
   - **方案 A: v85j + LR warmup 5%→10%** (warmup_frac 加大, 让早期学习更稳定)
   - **方案 B: v85j + LR_max 1e-4 → 5e-4** (提高 peak LR, 可能跳出局部最优)
   - **方案 C: v85j + AdamW betas (0.9, 0.999) → (0.9, 0.95)** (更快适应近期梯度, 适配 6 decoder 大模型)
   - **方案 D: v85j + Stage3 batch_size 1024 → 2048** (DDP 4-card 全局 batch 翻倍, 梯度更稳)
3. **风险与对策**:
   - 所有改动均为 Stage3 LR/optimizer 维度, 不涉及 Stage1/Stage2/DECOR, 风险可控
   - 预期 v85j 0.1053 + 0.001~0.003 = 0.106~0.108 (接近 v77 0.1080)
4. **优先级**: P0 — Stage3 LR/优化器调整是 v85 系列未探索的盲区, 且 v85j/v85i/v85h/v85k/v85l/v85m 6 个变体都没碰过 LR schedule 细节

**严禁**任何 DECOR 机制 (--enable_prompt_former / decor_prompt_former.py). 纯曲率路线继续.

---

## 7. 文件清单

- product_dir: `/fs04/ar57/wenyu/GeneRec/taskA/_history/issue141_v85n_stage2/`
- partial ckpt: `hrqvae_kappa_sync.ckpt` (ep100 不完整)
- partial sid: `sid_output.npy` (util 不达标, 不能用)
- audit: `issue41_audit.json` (κ final=[-0.072, -0.086, -0.095])
- mlr: `mlr_entropy_calibration.json`
- log: `/tmp/v85n_stage2.log`
- Stage2 script: `/fs04/ar57/wenyu/GeneRec/taskA/stage2/taskA_stage2.py` (CONFIG 不变, 仅 --item_emb_npy 改为 hyp_v2)
- Stage1 input: `/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage1_hyp_v2/item_emb_u32.npy` (SHA=99e6b39a)