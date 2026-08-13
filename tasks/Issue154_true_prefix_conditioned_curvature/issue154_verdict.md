# Issue #154 — True Prefix-conditioned Curvature Verdict

## 总结

| Gate | 判定 | 说明 |
|---|---|---|
| Gate 1 初始化与梯度通路 | **PASS (11/11)** | 初始 delta=0, A/B 距离/loss/SID 逐元素一致; fc1 Xavier 非零 (norm 5.8), fc2 零; step1 后 fc2 有梯度, step2 后 fc1/fc2 通路形成; prefix stop-grad 生效 |
| Gate 2 机制激活 | **FAIL → ROUTER_COLLAPSE** | delta_std / c_std / shuffle counterfactual 均未达阈值, 机制未激活 |
| Gate 3 曲率因果诊断 | SKIPPED | Gate 2 失败, spec 禁止进入 |
| Gate 4 完整 A/B 评估 | SKIPPED | 同上 |

## Gate 2 关键数据 (200 epoch 训练后, 全量 9922 商品)

| 指标 | 要求 | L1 | L2 |
|---|---|---|---|
| delta_std | > 1e-3 | 4.6e-4 ❌ | 1.7e-4 ❌ |
| c_effective_std | > 1e-4 | 4.5e-5 ❌ | 1.7e-5 ❌ |
| shuffle_fraction_dc>1e-4 | ≥ 10% | 11.2% ✅ | **0% ❌** |
| fc1/fc2 非零 | 必须 | ✅ (5.81 / 0.063) | ✅ (6.53 / 0.009) |
| boundary_hit | — | 0 (无饱和) | 0 (无饱和) |
| delta 范围 | — | [-0.0031, -0.0008] | [-0.0010, +0.0001] |

## 核心发现: #154 修复生效但机制仍不激活

**#154 修复验证 (Gate 1)**: 旧版 (Issue #147/152) 路由器 `fc1.weight=0` 全零初始化 → `relu(0)=0` → `fc2.weight` 梯度恒 0 → 路由器只学到 fc2.bias 常数偏置 (所有商品 delta≈1.45, "常数退化")。#154 改 `fc1` Xavier 非零初始化后, 梯度通路真实形成 (fc2.weight 从 0 学到 0.063/0.009), Gate 1 全部验证通过。

**但机制仍未激活 (Gate 2 FAIL)**:
- 200 epoch 后 delta 分化仅 1e-3 量级, c_effective_std 仅 1e-5 (要求 1e-4)
- L1 shuffle counterfactual 11.2% (勉强过线), **L2 完全 0%** — L2 路由器 (输入 [e0,e1] 64 维) 信号更弱
- SID 4-digit unique: A=9169 vs B=9168 — 曲率微小变化对 argmin assignment 几乎无影响

**根因**: 量化 loss 对 c 的梯度经 `sigmoid(θ+δ)` 与 Poincaré 距离函数两级衰减; per-item c 的 1e-4 变化不足以改变任何 codeword assignment, 路由器缺乏有效学习信号。这是机制性限制, 不是超参数问题 (R36 禁调参)。

## R18 4 维度对比

- **D1 spec**: fc1 Xavier 非零 + fc2 零初始化 (初始 delta=0 严格等价); 路由正则用未 detach delta; Gate 2 量化阈值
- **D2 实施**: PrefixRouter Xavier 修复 + `_last_delta_live` 正则 + gate2_mechanism.json 全量统计; control 保持 baseline A arm
- **D3 Gate 1 失败机制**: 首次 gate1 FAIL 因 gate1_precheck sys.path 冲突加载旧版全零 fc1 — 修复后 PASS; step1 fc1 梯度为 0 属预期 (fc2=0 时梯度不穿透)
- **D4 文献**: 延续 #147 QINCo (ICML 2024) 设计, 修复点来自 #147 失败根因分析

## 判定

**ROUTER_COLLAPSE** — 按 spec "Gate 2 失败判定为 ROUTER_COLLAPSE, 停止 Stage3/4" 立即停止, 不进入推荐评估。

结论: #154 成功诊断并修复了 #147 的常数退化 (fc1 全零) 问题, 但修复后路由器仍学不到足够的 per-prefix 曲率分化。prefix-conditioned curvature routing 在纯 Stage2 曲率路由 (Stage3 输入侧不变) 约束下无法激活机制 — 与 #147/152 INVALID 结论一致。此 lineage 关闭。

## 产物

- `gate1_initialization.json` (11/11 PASS)
- `control/stage2/{hrqvae_kappa_sync.ckpt, sid_output.npy, verdict.json, train_log.jsonl}`
- `treatment/stage2/{同上 + gate2_mechanism.json}`
- `issue154_verdict.json` (本文件机器可读版本)
