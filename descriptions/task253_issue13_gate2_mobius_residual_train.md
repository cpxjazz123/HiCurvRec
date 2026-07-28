# Task #253 — Issue #13 Gate 2: Möbius 残差算子实际训练 + Stage 4 eval (大型 GPU)

## 来源
- GitHub Issue #13 (2026-07-28): [Escape Route 3] 残差算子几何一致性审计
- 承接 Task #248 (Gate 0 PASS: 残差算子是欧式减法) + Task #249 (Gate 1 PASS: 距离替换确实改变分配)
- 主文献: Piękos P. et al. *Hyperbolic Residual Quantization*, arXiv:2505.12404v1

## 任务目的

Issue #13 §阶段闸门 Gate 2: 实际训练 + Stage 4 eval, 验证 "残差算子 Möbius 化" 是否能在 Stage 4 端到端击败 R@10 = 0.1020.

通过条件:
- R@10 > 0.1020 (HG-Rec baseline), 或
- 显著接近/超过 0.1058 (仓库当前最优, phonism), 或
- 比 PC κ 版本 (verdicts/task225 R@10=0.0938) 显著提升

硬停止:
- 训练不收敛 (loss 不下降 / NaN)
- Stage 4 R@10 < 0.05 (彻底崩溃)
- 改动引发 Stage 1 训练崩溃 (跟 task180 引入 mode collapse 同类风险)

## 决策 (R11.4 critical decision, dry-run 报告)
**改动位置**: `HG-Rec/model/utils.py:1795` 单点 `residual = residual - x_res`

**改动内容**:
```python
# 原 (欧式减法):
x_res, loss, indices = quantizer(residual, use_sk=use_sk)
residual = residual - x_res  # ← utils.py:1795

# 新 (Möbius + 欧式 混合):
x_res, loss, indices = quantizer(residual, use_sk=use_sk)
# product_manifold: hyp part 用 Möbius 加法 (mobius_add(-x_res_hyp, residual_hyp, c))
#                   euc part 保持欧式减法 (跟 hyp_dist 配)
if hasattr(quantizer, 'c'):
    residual_hyp = mobius_add(-x_res[:, :hyp_dim], residual[:, :hyp_dim], quantizer.c)
    residual_euc = residual[:, hyp_dim:] - x_res[:, hyp_dim:]
    residual = torch.cat([residual_hyp, residual_euc], dim=-1)
else:
    residual = residual - x_res  # 退回欧式 (防御)
```

**风险 (R11.4 critical decision)**:
- 改动上游 `utils.py:1795` 不可逆, 必须 backup 后改
- product_manifold 训练中 `proj_to_ball` 守 hyp 部分在 Poincaré 球内, Möbius 加法也应在球内守恒 (utils.py:34 mobius_add 实现已含 proj_to_ball logic)
- Gate 2 短训 50 epoch = ~3 hours wall, GPU 0/3 空闲可启动
- 不用 task222 ep29 ckpt 微调 (避免跟历史混淆), 重新跑 Stage 1 短训
- 必须配套 R12 强制 ckpt 保存 (训练中每 N 步保一次)

## 产物
- `HG-Rec/model/utils.py:1795` patch (Möbius 残差)
- `scripts/task253_issue13_gate2_mobius_residual_train.sh` launcher
- `products/task253/mobius_residual/<ts>/best_ckpt.pth`
- `verdicts/task253_issue13_gate2_mobius_residual_result.md`

## 状态
- 大量 GPU + 改上游
- ~3 hours wall (50 epoch 短训, R12 强制 ckpt)
- 启用 GPU 0 (避开 1/2 被 Task #243 占)
- 训练完成后单独 Stage 4 eval (~5 min, 套用 task233 §8 inline template)