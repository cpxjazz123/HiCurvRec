# v19_hab_delta_curvature R38 早停回退 verdict (2026-08-09)

## R38 决策行
**v19 mid-training regress, v19 best_valid=0.1239 < v18 best_valid=0.1256 (-0.0017), 连续 49 epoch 无改进 (Ep115 后饱和), 立即 kill + 触发 R37 回退至 v18 重新创新**

## 用户硬约束 (R35 + R36 + R37 + R38)
- R35: 单 ckpt + beam_search=20, 禁 Borda/ensemble
- R36: 禁调参 (LR/dropout/ls/WD sweep), **必走曲率机制改善**
- R37: 新版本 test_R@10 < 上版本 → 必须回退至上版本
- R38: 训练期 mid-training regress → 立即 kill + 分析 + 触发 R37

## 实施 (R36 合规 — 改善曲率机制)
- Stage 3: v18 recipe (HAB + LR=1e-3 + branch_curvature) + **启用 `--hab_delta_curvature`**
  - v18: 单一 κ per layer + branch_curvature (按 SID frequency 分桶)
  - v19: v18 + ΔD 距离度量 (Dbar = ΔD/median, 而非绝对距离 D)
  - 理论动机: 用相对距离差替代绝对距离, 改变 HAB 曲率响应机制
- Stage 4: `tasks/v19_hab_delta_curvature/stage4_beam20.py`, beam_size=20 (未执行, 因 R38 早停)

## Stage 3 训练 (R38 早停)
- 启动 22:52:45 → R38 触发 23:28 (kill) → 实际跑 ~35 min
- **mid-training 信号**:
  - Ep115 best valid_R@10=**0.1239**
  - 之后连续 49 epoch (Ep116-Ep164) 无改进, loss 2.59-2.61 平台
  - early stop 24/30 持续上升
- **v18 对比**:
  - v18 best valid=0.1256 (Ep184)
  - v19 best=0.1239 落后 **-0.0017** (-1.4%)
  - v19 曲线已饱和, 极不可能反超 v18
- **R38 触发条件** (满足 1+2):
  1. ✅ best valid 落后 vN-1 (0.1239 < 0.1256)
  2. ✅ 连续 ≥30 epoch 无改进 (实际 49 epoch)
- **R38 执行**: pkill -9 PID 1658081, 1658030, 1658082-1658084 (rank0 + launcher + 其他 ranks)
- **未跑 stage4** (R38 早停, 仅记录 ckpt 留档, 不作下版本起点)

## R38+R37 综合决策
- v19 (HAB ΔD mode + branch_curvature) 中训练已确定 regress
- **R37 触发**: v19 视为失败 lineage, 终止该方向
- **回退基线**: v18_branch_curvature test_R@10=**0.1011** (单 ckpt + beam=20, 当前最优)
- **新版本起点**: 必须从 v18 完全配置 + 仅新加机制开始, 禁在 v19 失败品上叠加

## R35+R36+R37+R38 严守
- R35: 单 ckpt + beam=20 ✓ (此 verdict 决策阶段, stage4 未跑)
- R36: HAB ΔD 是曲率机制改善 ✓ (R36 合规)
- R37: 失败即终止 lineage ✓ (v19 标记为 NO-GO, 不作下版本起点)
- R38: mid-training regress 立即 kill ✓ (早停节省 ~9 min GPU 时间)

## 产物路径 (留作记录, 不作下版本起点)
- ckpt (Ep115 best): `/fs04/ar57/wenyu/GeneRec/taskA/_history/v19_hab_delta_curvature_stage3/HG_Rec_best.pth`
- train.log: `/fs04/ar57/wenyu/GeneRec/taskA/_history/v19_hab_delta_curvature_stage3/train.log`
- stage3 脚本: `tasks/v19_hab_delta_curvature/stage3.py`
- stage4 脚本: `tasks/v19_hab_delta_curvature/stage4_beam20.py` (未运行)

## 下一步 (R38 + R37)
- 当前最优基线: **v18_branch_curvature test_R@10=0.1011**
- 新版本方向 (R36 曲率机制, R37 从 v18 重新开始):
  - v20_branch_n5: branch_curvature n_buckets=3→5 (更细粒度分桶)
  - v20_branch_wider: branch_curvature λ_mult [0.5, 1.0, 1.5] (更激进曲率差异)
  - v20_kappa_learnable_stage3: Stage 3 时 Stage 2 κ_l 变可学习 (更激进曲率改善)
  - v20_minkowski: 替换 Poincaré 为 Minkowski 曲率机制
  - v20_lorentz: 替换为 Lorentz 曲率机制
- 选择标准: 哪个最有理论依据 → 立即从 v18 配置复用 + 仅新加机制
