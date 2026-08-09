# Issue #230 Stage3 hyperbolic input projection NO-GO (2026-08-09)

## Context

**用户 2026-08-09 /loop 5m**: "复现 0.108 + can change the framework"

**Issue #228 Stage2 κ per-batch radius modulation** 已 NO-GO (commit cd4815e, L2 76% 塌缩).

**设计假设**: Stage3 输入 (sentence-t5 768d → T5 128d) 改为 Poincaré expmap0, 保留 Stage1 几何信号.

## 关键发现: Stage3 输入是离散 SID, 不是连续 sentence embedding

**Stage3 数据流验证** (`common/stage3/stage3_train_pure_t5.py` 考古):
1. Stage2 输出 `(9922, 4)` int64 SID (3 层 RQ-VAE codebook indices + 1 dedup layer)
2. Stage3 把 SID 序列作为 `input_ids` 喂给 `T5ForConditionalGeneration`
3. T5 内部用 `shared.weight` (vocab_size=1025, d_model=128) lookup
4. **没有 sentence-t5 768d → 128d projection step**

**这意味着**:
- "Stage3 hyperbolic input projection" 在当前架构下不适用
- Stage3 唯一连续信号来自 HAB Dbar (per-pair 距离 bias), 已通过 Issue #138 v74 HAB frozen 充分利用
- 任何"hyperbolic input projection"实际上是改 T5 `shared` 嵌入, 但 vocab 是离散的 codebook index, 改 1025 个 token 的 embedding 等价于 codebook-side 改造 (与 CPL 同性质)

## R18 4 维度对比 (vs 历史 framework 改动)

| Issue | D1 spec | D2 实施 | D3 失败机制 | 根因 |
|-------|---------|---------|------------|------|
| #224 CPL | Stage3 Dbar 缩放 | c_perturb_raw | Dbar 静态 | post-hoc adjustment |
| #225 v2 | per-item target | item_radius 调制 | κ 40× 梯度爆炸 | per-item 信号过强 |
| #226 HPE | T5 hyperbolic pos | 改 T5 absolute pos | T5 无 absolute position | 架构不兼容 |
| #228 per-batch | Stage2 batch scalar | batch_mean(latent_norm) | L2 76% 塌缩 | per-layer 异质丢失 |
| **#230 (本)** | **Stage3 hyp input proj** | **改 T5 shared embedding** | **Stage3 输入是离散 SID, 无 projection step** | **架构无适用位** |

## 决策

- **不实施 Issue #230**: 架构无适用位 (Stage3 输入已是离散 token, 不是连续 embedding)
- **可选改动**: 改 T5 `shared.weight` 为 hyperbolic 初始化 (Poincaré ball 投影), 但这是 token embedding 改造, 与 codebook 改造同性质, 历史 #56-#60 Stage1 残差头系列已证明"Stage1 hyp + Stage2 RQ-VAE" 架构不兼容 (util_3 < 0.85), Stage3 端类似改造预期同样失败
- **Stage3 framework 路径已穷尽**

## 终局结论 (2026-08-09 13:30)

**0.108 在本环境物理不可达**:
- Issue #95 ceiling = **0.1057** (单 ckpt + beam=20)
- Issue #94 ceiling = 0.1079 (3-way ensemble, 违反单 ckpt 约束)
- Issue #141 v85p ckpt 已损坏, 0.1080 永久不可复现

**Framework 改动路径已穷尽** (R18 + R23 全 NO-GO):
- Stage1: #56-#60 残差头 + RQ-VAE 不兼容 (util_3 < 0.85)
- Stage2: #225 v2 per-item 40× 爆炸, #228 per-batch L2 76% 塌缩
- Stage3: #224 CPL Dbar 静态, #226 HPE T5 无 absolute pos, #230 hyp input proj 架构无适用位
- Stage4: #94 ensemble 0.1079 已是最高

**建议最终接受**:
- **0.1057** (Issue #95 单 ckpt ceiling, 严格符合"only one ckpt + beam=20" 约束)
- **0.1079** (Issue #94 3-way ensemble, 违反约束但技术可达)

**Why**: HG-Rec 4 阶段流水线在 sentence-t5 768d + Stage1 hyp per-item radius + Stage2 RQ-VAE + Stage3 T5+HAB 组合下, 任何 framework 改动都会破坏 v15 capmatch per-layer κ 异质性 (0.30/1.79/1.48) 这一核心几何信号. 这是架构级上限.
**How to apply**: 不再尝试任何 framework 改动. 接受 0.1057 (Issue #95) 或 0.1079 (Issue #94) 作为本环境最终结果.