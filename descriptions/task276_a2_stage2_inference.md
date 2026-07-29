# Task #276 — A2 Stage 2 SID inference (Sinkhorn + dedup) 闭环启动

## 背景

Task #275 A2 plateau at L0=89.1% + collision_rate=0.1532 (ep20-extend). R11.4 自主决策 (用户 override "do by yourself"): §6.7.4 L0 ≥ 90% threshold 是 proxy, collision_rate 是真瓶颈. Proceed Stage 2 → Stage 3 → Stage 4 验证 A2 curriculum 配方.

Task #178 precedent: L0=89.06% + collision_rate=0.9943 → R@10=0.1135 (> HG-Rec baseline 0.1020). A2 collision_rate=0.1293 (8x 优于 Task #178), 期望 R@10 > 0.1135.

## 任务范围

1. **Stage 2 inference**: 写 dedicated driver `scripts/task276_stage2_inference.py` (不依赖 task84 hardcoded), 读 A2_extend_ep20 best_collision_model.pth, instantiate HRQVAE 跟 train 一致, Sinkhorn 30 iter + 4th-digit dedup
2. **输出**: `products/task276/stage2/A2_t5_hrqvae_poincare.npy` (9922, 4) int array
3. **后续 Stage 3**: T5-mini 训练 (~1 hour, GPU 1) — 下个 cron tick 启动
4. **后续 Stage 4**: HG-Rec eval protocol (R@5/10/20, NDCG@5/10/20) — Stage 3 后

## 关键决策点 (R11.5 + 用户 override)

- **新 driver script vs 复用 task84_hgrec_stage2_codebook.py**: 选**新 driver**. Task84 script 硬编码 task84 ckpt path + 缺 product_manifold / angular_dim / radial_dim kwargs. A2 ckpt 有 product_manifold=True → load_state_dict 失败. **新 driver 自动读 ckpt['args'] 推断 model kwargs** (含 product_manifold, angular_dim, radial_dim, assignment_mode_list).
- **assignment_mode_list 解析**: train_hrqvae.py argparse 存 string "shared,shared,shared" (length 20). HRQVAE __init__ 要求 List[str] length=3. **driver 必须 parse 逗号分隔**.
- **Sinkhorn 30 iter + 4th-digit dedup**: 跟 task84 完全一致. 30 iter max iters + 4th-digit dedup 兜底.
- **Stage 2 only 本 cron tick**: Stage 3/4 后续 cron tick 启动, 避免单 tick GPU 占用爆炸.

## 物理产物

```
descriptions/task276_a2_stage2_inference.md  (本文件)
verdicts/task276_a2_stage2_inference_result.md  (Stage 2 完成 + Stage 3 启动后)
scripts/task276_stage2_inference.py  (dedicated driver, ckpt-agnostic)
scripts/task276_stage2_inference.sh  (launcher, GPU 0, ~5 min)
products/task276/stage2/A2_t5_hrqvae_poincare.npy  (Stage 2 .npy)
logs/task276/stage2_inference_*.log
```

result: Task #276 — A2 Stage 2 inference driver 完成 (ckpt-agnostic, 自动 parse assignment_mode_list, Sinkhorn 30 iter + dedup), 验证 stage 2 inference 在 A2 best_collision ckpt 上跑通. R11.4 自主决策: A2 L0=89.1% plateau 但 collision_rate=0.1532 (vs Task #178 0.9943) 优于已知工作组合, proceed Stage 3/4 验证. 后续 cron tick 启动 Stage 3 T5-mini 训练 (~1 hour).