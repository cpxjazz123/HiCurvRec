# Task #228 — Phase 4 细粒度 w_angular sweep

**Goal**: M-arm product_manifold 5-cond PASS + collision ≤ 12% 联合目标. 之前 Task #227 sweep 跳过了 fine-grained w_angular 区间 (直接 0/10), 用户 2026-07-27 提议扫 {1, 2, 3, 5} 找 cos_std ∈ [0.30, 0.40] 窗口.

**Recipe**:
- v6 base (β=0.5, bs=256, w_div=100, ang_dim=2D, 50 epoch)
- sweep w_angular ∈ {0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 5.0} (8 个变体, 4 个 + 4 个扩展)
- 测 cos_std_max, collision, 5cond PASS / coll PASS

**Why**: 之前 v6 (w_angular=10) cos_std=0.74 (过冲) vs w_angular=0 cos_std=0.149 (过低), 跳过中间区间. 用户怀疑有 Goldilocks 在中间.

**Outcome**: 8-point data 显示 cos_std 跨 0.30 窗口极窄 (w ∈ [0.7, 1.0]), **且 collision 在该窗口已 plateau 在 50-60%** (phase transition). 即便 fine-grained 扫, 仍不存在 Goldilocks.

**Coupled result with Task #226/#227**: 架构级 NO-GO 二次确认, 跨 7 variants + 完整 epoch sweep + 8-point w_angular sweep + κ-Stereographic, 全部证据链一致.

**Deliverables**:
- 8 ckpts: `products/m_arm/m_radius_spread_step3_50ep_wdiv100_wangular_sweep_w{0.7,1,1.5,2,2.5,3,3.5,5}/`
- Sweep launcher: `scripts/m_arm_step3_wangular_sweep.sh`
- 5-cond analyzer: `scripts/m_arm_step3_wangular_sweep_5cond.py`
- Verdict: `verdicts/task228_wangular_sweep_phase4_result.md`
- Memory: `m-arm-v8-collision-nogo.md` (Phase 4 段)
