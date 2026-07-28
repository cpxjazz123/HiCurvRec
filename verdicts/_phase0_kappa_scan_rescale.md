# Phase 0 重测 — 范数放大下 κ 几何确实激活 (2026-07-27)

## 关键发现 (R11.3 选项 C)

E1 normcap latent ‖z‖≈0.13 (基准). 手动放大 latent + codebook 范数 (scale=5/10/20) 模拟无 NormCap 状态.

| scale | ‖z‖ | 球面侧激活条件 | agreement | usage |
|---|---|---|---|---|
| 1x | 0.13 | 全部 100% 跟欧式一致 | — | 0.04-0.25 |
| **5x** | **0.65** | **κ=+10 → agreement 0.22** ✅ 激活 | 0.22 | 0.39 |
| 10x | 1.31 | κ=+3 → agreement 0.032 ❗ 全反转 | 0.03 | 0.39 |
| 20x | 2.62 | κ≥+1 → agreement ≈ 0 ❗ 全部反转 | 0.0 | 0.04-0.08 |

## 解读

1. **球面 κ 几何确实存在**: 大范数 + 大 κ 时 agreement 从 1.00 跌到 0.22, 跟欧式明显不同
2. **但用户判据 "agreement 60-90%" 在我的扫描里没出现** — 只有两个极端: (a) 完全跟欧式一致 (几何未激活) (b) 完全反转 (几何过度激活)
3. **usage 全部 ≤ 39%**: 跟 κ 几何激活同时出现, 永远伴随码字坍缩到几个 cluster

## 模式总结

```
大范数 + 小 κ   → agreement 100%, usage 8-25% (欧式距离主导, 跟 E1 baseline 等效)
大范数 + 大 κ   → agreement 0-22%, usage 4-39% (几何激活 + usage 崩)
小范数 (NormCap) → 全部 100% agreement (几何被范数尺度淹没)
```

**没有甜区** — 几何激活跟码字坍缩是同一个现象的两种表现.

## 候选下一步 (R11.4 关键决策, 等用户决定)

| 选项 | 含义 | 行动 |
|---|---|---|
| A | 关闭球面线, 转写作 | 写综合 verdict (J-plan NO-GO + Phase 0 NO-GO + 收尾) |
| B | 取消 NormCap 重训 E1 baseline, 跑 Phase 0 真实几何测试 (‖z‖≈0.5-1.0 自然落点) | 6 h 训新 ckpt, 30 min Phase 0 |
| C | 细扫 κ ∈ {5..9} × scale ∈ {3..6}, 看能否找到 "agreement 60-90% AND usage ≥ 80%" 甜区 | 5 min CPU |
| D | 用另一个非 NormCap ckpt (e.g. task220 hrqvae_pck) 跑 Phase 0 | 5 min CPU (不需要重训) |

按 R11.3 推荐: **D 优先** (5 min, 不重训). task220 是 hrqvae_pck (Per-Codeword κ), 不带 NormCap, latent 范数可能天然在 0.5-1.5. 若 task220 latent 跟 Phase 0 重测的 5x-10x 状态匹配, 球面侧几何能激活, 可进入 Phase 1. 若 task220 也小范数, 关闭球面线.

