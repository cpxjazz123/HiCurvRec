# Task #273 — Task #178 R@10=0.1135 §6.7.4 stop-loss (i) 复核

## 背景

Task #272 inventory 列出 Task #178 (fixed baseline recipe, 200 epoch) R@10=0.1135 是当前**最高复现 R@10**, 但**§6.7.4 stop-loss (i) 待复核** (L0 utilization 未直接测量).

Task #178 ckpt args 关键字段:
- `bn=True` (有 BN layers, verifier 之前没传 → shape 不匹配 crash)
- `layers=[512, 256, 128, 64]` (encoder down)
- `num_emb_list=[64, 128, 256]` (L0=64, L1=128, L2=256)
- `e_dim=32` (不是 36 也不是默认 64)
- `sk_epsilons=[0.003, 0.003, 0.003]` (baseline sinkhorn)
- `beta=0.5` (baseline commitment loss)

## 任务范围

1. 补丁 task263 verifier 支持 `bn` 参数 (新 ckpt 必需)
2. 直接测 task178 best_collision_model.pth 的 L0/L1/L2 utilization
3. §6.7.4 stop-loss (i) 判定: L0 ≥ 90%? 是 → 通过; 否 → 越闸
4. 0 GPU 启动 GPU (~30 sec measurement)

## 关键决策点 (R11.3)

- **共享 verifier 兼容补丁**: 修改 task263 verifier 加 `bn=ckpt_args.get("bn", False)`. 默认 False 不影响 task253 (#253 ckpt bn=False 已 default 兼容). R12 严格存.
- **判定 L0=89.06%** — 这正好卡 §6.7.4 阈值下, 需要在 verdict 明确"差 0.94pp, 严格违规但数值极 borderline".

## 物理产物

```
verdicts/task273_task178_utilization_audit_result.md
descriptions/task273_task178_utilization_audit.md
products/task273/task178_bestcollision_utilization.json
scripts/task263_issue17_gate2_task253_direct_utilization_meas.py  (patched — bn=False default)
```

result: Task #273 — task178 L0 utilization 直接测量 (89.06%) + §6.7.4 stop-loss (i) borderline 判定. L1/L2 严重坍缩 (0.78% / 0.39% = 单码字). 高 R@10=0.1135 来源: L0 borderline 接近 90% 而非 §6.7.4 严格达标.
