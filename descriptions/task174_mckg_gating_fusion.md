# Task #174 — MCKG 门控融合 D 臂 (用户 design §5) [Stage 1/2/3/4]

> **任务目的**: 用户 11:55 显式决策 "继续 κ-Stereo 路径: 启用 MCKG 门控融合 (您设计 §5 D 臂)". κ-Stereo 路径 8 个变体穷尽后 (#165-#172, C3 NO-GO), 这是 user design 文档 §3/§5 唯一 unexplored D 臂.

> **完成日期**: (in progress)
> **状态**: 🟡 设计 Phase + 等 GPU 释放 (#169/#170/#171/#172 占用 GPU 0/1/2/3)

---

## 1. 背景

承接 κ-Stereo 8 变体 (#165-#172) C3 (下游 > 0.1058) NO-GO. 用户 design 文档 §5 表 explicit unexplored D 臂: **MCKG 门控融合**, M=2 components + 门控网络融合.

按用户原 hard constraint "使用κ-stereographic 距离公式", D 臂严格遵守 (κ_m 可学习 + κ-Stereographic 测地距离), 只改变距离合成机制.

**Why D 臂不是简单重复 A 臂 (M=1)**:
- A 臂 (Task #89): M=1, 单分量, 18/18 (layer, κ_m) = 0 → κ_m 无空间
- B/C 臂: M=2/3 baseline 合成 (sqrt sum sq), 但 κ_m 同 A → 0
- **D 臂 (本任务)**: M=2 + 门控 — 即使 κ_m = 0, 门控网络 w_m(x,c) 仍能学到 "哪些 component 维度更重要", 这是 A 臂验证不到的变量

## 2. 实验设计

**变量**: M=2 components + 门控网络 `w_m = softmax(MLP([x_1, x_2]))` 融合距离.

**距离函数** (用户 design §3 advanced 档):
```python
dist(x, c) = Σ_m w_m(x, c) · d_{κ_m}(x_m, c_m)
# x ∈ R^n split into x_1, x_2 ∈ R^{n/2}
# d_{κ_m} = κ-Stereographic geodesic distance (Berman-Metzler 2020)
# w_m = softmax(MLP(x_1 ⊕ x_2))_m ∈ [0,1], Σ_m w_m = 1
```

**保持不变**:
- κ-Stereographic Phase A/B (kappa_freeze_epochs=100, lr_theta=1e-5, theta_init=[0,0,0] for 2 components × 3 layers = 6 values)
- codebook [32, 64, 256] e_dim=32 split into M=2 components of 16-dim each
- layers 512 256 128
- Stage 2/3/4 沿用现有 pipeline (T5-mini 9.18M)
- seed=42

**启动命令** (待 GPU 释放 + D 臂实现完成):
```bash
bash scripts/task174_stage1_mckg_gating.sh       # 修改 task89 launcher 加 M=2 + gating
bash scripts/task174_stage2_codebook.sh
bash scripts/task174_t5mini_mckg_gating_stage3.sh
bash scripts/task174_t5mini_mckg_gating_stage4_eval.sh
```

## 3. 决策触发(vs baseline R@10=0.1058)

| 指标条件 | 结果指标 | 决策 |
|----------|----------|------|
| test R@10 > 0.1058 | > baseline | ✅ PASS — D 臂 gating + κ-Stereo 协同达成 C3 |
| 0.10 ≤ test R@10 ≤ 0.1058 (≈ #166) | 持平 | 🟡 gating 加 κ 没增益 |
| test R@10 < 0.10 | < baseline -5% | ⛔ NO-GO — κ + gating 都无法补足 downstream deficit |

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| D 臂实现 (fork hrqvae_free_curv.py + 加 M=2 split + 加 gating network) | ~2-3h |
| Stage 1 Phase A/B 200 epoch | ~30 min |
| Stage 2 SID inference | ~1 min |
| Stage 3 T5-mini 200 epoch | ~30 min (early stop ~20) |
| Stage 4 eval | ~2 min |
| 总计 | ~3-4 h |

## 5. 实施路线

**Step 1**: Fork `HG-Rec/model/hrqvae_free_curv.py` → `hrqvae_mckg_gating.py`
**Step 2**: 修改 `FreeCurvVectorQuantization._per_component_dist_sq` (line 240):
  - 当前已支持 M>=1, 但默认 M=1
  - 加 M=2 split: x_full (B, e_dim=32) → x_comp (B, M=2, e_dim/2=16)
  - 加 gating MLP: w = softmax(Linear(32 → 32) → ReLU → Linear(32 → M=2))
  - dist = Σ_m w_m · d_{κ_m}(x_m, c_m)
**Step 3**: task89_stage1 launcher 加 `--num_components 2` + `--gating_network enabled` args
**Step 4**: theta_init_list 长度: 2 components × 3 layers = 6 theta values (vs current 3)
**Step 5**: Stage 2/3/4 镜像 #169-#172 pattern

## 6. 风险与缓解

**风险 1**: κ_m 全部 → 0 跟 A 臂一样 NO-GO (即使 gating 学习有意义的 w_m, κ_m 仍是 0 没几何意义) → 接受为 "门控机制 vs 几何" disambiguation 实验
**风险 2**: 实现 bug 引入训练 instability (gating network 软门控可能引入梯度爆炸) → Stage 1 加 log_interval=20 监控 grad norm, 用 grad clip=1.0
**风险 3**: 跟 Task #89 A 臂 ROA 重叠 (一个 component 子集) — 但 #89 没用门控, 是纯 disentangle, 严格不同

## 7. 完成度跟踪

- [x] R10 + user decision 同步记录
- [x] task #167 in TaskList
- [ ] Fork hrqvae_mckg_gating.py
- [ ] 修改 _per_component_dist_sq 加 M=2 + gating
- [ ] task174 launcher scripts
- [ ] 等 GPU 0/1/2/3 释放 (现被 #169-#172 占用)
- [ ] Stage 1 launch + Phase A/B 训练
- [ ] Stage 2 codebook
- [ ] Stage 3 T5-mini
- [ ] Stage 4 test eval
- [ ] Verdicts/task174_*.md
- [ ] #163 synthesis verdict 加 D 臂节

## 8. R11.3 决策明示

**选了** D 臂 (MCKG 门控融合, M=2 + 门控网络).
**为什么**: (a) user 11:55 显式决策; (b) 是 user design 文档 §3/§5 D 唯一 unexplored 臂; (c) 即使 κ_m=0, 门控机制独立于几何贡献可被检验, 不跟 A 臂 (Task #89) 重叠.
**备选**: (a) A 臂重跑 (#89 已 NO-GO) (b) B/C 臂 baseline 合成 (跟 A 臂 κ_m=0 NO-GO) (c) vanilla baseline (违反 C1 hard constraint, REVERTED).
**ROI 评估**: 中等 — 2-3h 实施成本 + 30min 训练, 失败也是 valuable 证据 (κ-Stereo + gating 都不 work → κ 路径彻底结束).
