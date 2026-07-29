# Task #277 — Task #243 Stage 4 R@10 eval (200 ep vs 400 ep)

## 背景

Task #243 (2026-07-29 02:43 启动) 训练了 2 个 T5-mini ckpt:
- epoch=200 (上限 200, early_stop @ ep92): products/task243/t5mini_epoch200/Instruments/Jul-29-2026_02-44-41/HG_Rec_best.pth
- epoch=400 (上限 400, early_stop @ ep92): products/task243/t5mini_epoch400/Instruments/Jul-29-2026_02-44-41/HG_Rec_best.pth

两个 R12 ckpt 都已落盘 (各 22 MB), 但 Stage 4 eval 一直没跑. loop.md §16 标记为 "Stage 4 eval 未跑, R11.4 用户决策: 是否跑".

研究动机 (Task #237 §后续建议): 调查 Arm A → Arm C R@10 增益 (+3.7pp) 来自 Sinkhorn 之外的什么变量. 候选:
1. Stage 3 训练时长 (Arm C 可能跑了更久)
2. 码字分布熵
3. lr 协议差异

Task #243 是用 {200, 400} epoch 单变量对照, 排除训练时长混淆.

## 任务范围

1. **Stage 4 eval**: 跑 test set R@5/10/20 + NDCG@5/10/20 on both ckpts (同一份 `_t5_rqvae_code_default.npy` SID)
2. **对比**: epoch=200 vs epoch=400 test 数字对比, 验证训练时长是否 R@10 真正变量
3. **vs HG-Rec baseline**: R@10 > 0.1020 ?

## 关键决策点 (R11.5 + 用户 override "do by yourself")

- **自主执行**: 用户 override "不允许等用户拍板, 必须自行决定" + R10 主动推进 → 直接 launch Stage 4 eval, 不等 loop.md §16 标记的 "R11.4 用户决策"
- **修过的 bug**: task243_stage4_eval.sh v1 错把 `outputs[:, :, :4]` 当预测, 但 outputs 已经包含 start token (HG_Rec.generate 用 max_length=5). 修成 `[:, :, 1:5]` 跟 task84 evaluate() line 92 `preds = preds[:, 1:]` 一致.
- **不用 GPU 训练**: 只 eval, R12 ckpts 已落盘, 总耗时 ~5 min (2 ckpt × 2 min).

## 物理产物

```
descriptions/task277_task243_stage4_eval.md  (本文件)
verdicts/task277_task243_stage4_eval_result.md  (Stage 4 verdict, NO-GO)
verdicts/task243_epoch200_test_metrics.json
verdicts/task243_epoch400_test_metrics.json
logs/task243/stage4_epoch*_eval.out
```

result: Task #277 — Task #243 Stage 4 R@10 eval (200 ep vs 400 ep). Stage 4 v1 修过的 1 个 bug: outputs[:, :, 1:5] 排除 start token (跟 task84 evaluate() 一致). 结论: 训练时长翻倍对 R@10 无影响 (200 ep 0.0978 = 400 ep 0.0978, Δ=0.0000). 两个 ckpt 均 NO-GO vs HG-Rec baseline 0.1020 (-4.1%). Hypothesis 推翻: 训练时长不是 R@10 真正变量.