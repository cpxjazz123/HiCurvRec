# Task #349 — Issue #62 Gate 1 Arm D Stage 1 NO-GO (Phase 0 mode collapse)

## 来源

承接 Issue #62 (owner 显式 opened, 2026-07-30 16:07) §Gate 1 Arm D spec (owner comment 2026-07-30T16:44:54Z):
- Stage 1 = #30 per-layer Codebook Transforms + #43 HypPreEncoder + K0=256 (Arm D)
- 1000 epoch batch=1024 + lr=1e-3 AdamW + sk_eps=0.0 (Sinkhorn DISABLED)
- --num_emb_list 256 128 256 (K0=256)
- --hyp_c 0.74 (Issue #43 HypPreEncoder)
- --radius_list 0.1 1.0 10.0 --scale_list 2.0 2.0 2.0 (Issue #30)

承接 task348 Gate 0 (commit 8620258, 5/5 PASS) + task344 Issue #62 description.

## 实施 spec

按 owner 完整 spec 启动 Arm D Stage 1, GPU 0:
```bash
nohup bash scripts/task349_issue62_gate1_armd_stage1_train.sh
# num_emb_list 256 128 256 (K0=256)
# num_epochs 1000 --batch_size 1024 --lr 1e-3 --sk_eps 0.0
# hyp_c 0.74, radius_list 0.1 1.0 10.0, scale_list 2.0 2.0 2.0
# beta 0.25 --save_limit 1 --seed 42
```

## 实证结果

USAGE-KILL @ epoch 30 (3.5 min 运行后):
- collision_rate=0.9999 (99.99% items 撞同一码字)
- L0: usage=0.4% (1/256)
- L1: usage=0.8% (1/128)
- L2: usage=0.4% (1/256)
- r_std=0.0000 (per-layer 几何变换未散开)
- recon_loss 0.0021 (卡死)

## 根因诊断 (Phase 0 Mode Collapse)

跟 task178/180/231/242/299 同模式:
- K0=256 + Sinkhorn OFF (sk_eps=0) + 无 κ-decouple + 1000 epoch = 1-码字坍缩
- 跟 task287 verdict §NO-GO 根因综合一致: 大 K + Sinkhorn OFF + 无 κ-decouple = mode collapse
- Arm D 失败 = owner spec 在 baseline recipe 内部 NO-GO, 不改变 #30+#43 联合 ablation 命题本身的潜力

## 决策

Arm D NO-GO 收口 ✅ (verdict JSON + MD 已写).
下一步: Arm C 启动 (#30 + #43 联合, K=[64,128,256] baseline + Sinkhorn ON, 1000 epoch batch=1024 lr=1e-3).
- Task #350 创建
- scripts/task350_issue62_gate1_armc_stage1_train.py + .sh 已写
- 决策阈值: 若 Arm C 也 USAGE-KILL = Issue #62 整体 NO-GO; 若 Arm C PASS Gate 1 = 进入 Stage 2 Sinkhorn + Stage 3 T5 + Stage 4 R@10 eval

## 物理产物

| 类型 | 路径 |
|------|------|
| Trainer wrapper | `scripts/task349_issue62_gate1_armd_stage1_train.py` |
| Launcher | `scripts/task349_issue62_gate1_armd_stage1_train.sh` |
| 训练日志 | `logs/task349/stage1_gate1_armd_20260731_114059.log` |
| Verdict JSON | `verdicts/task349_issue62_armd_stage1_nogo.json` |
| Verdict markdown | `verdicts/task349_issue62_armd_stage1_nogo.md` |