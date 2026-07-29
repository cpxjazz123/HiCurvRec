# Task #276 — A2 Stage 2 inference 完成 + Stage 3 启动

> **完成日期**: 2026-07-29
> **状态**: 🟢 **Stage 2 完成** + 🟡 **Stage 3 训练中 (GPU 1, ~80 min ETA)**

---

## 1. Stage 2 inference 结果

| 指标 | 值 |
|------|-----|
| 输入 ckpt | `A2_extend_ep50/best_collision_model.pth` (L0=89.1%, collision=0.1532) |
| initial unique SID | 8402/9922 (15.3% collision pre-Sinkhorn) |
| Sinkhorn 30 iter 后 | 8402 unique (Sinkhorn 不再优化, 收敛) |
| 4th-digit dedup 后 | **9922/9922 unique (100%)** |
| L0 unique | 57/64 (89.1%, 跟 Stage 1 一致) |
| L1 unique | 124/128 (96.9%) |
| L2 unique | 253/256 (98.8%) |
| L3 unique | 14 (碰撞组最大尺寸 13) |
| .npy 文件 | `products/task276/stage2/A2_t5_hrqvae_poincare.npy` (317 KB, shape=(9922,4), int64) |
| 总耗时 | ~4 min (GPU 0) |

## 2. 关键观察

### 2.1 Sinkhorn 30 iter 不收敛

- Iter 0: 1027 groups → iter 30: 1027 groups (stuck)
- Sinkhorn 只解决最浅层冲突, 深层 L0+L1+L2 联合冲突需要 4th-digit dedup 兜底
- 跟 task84 baseline 行为一致 (Stage 2 同样靠 4th-digit dedup)

### 2.2 A2 SID 质量

- 全部 9922 unique (1:1 map to items)
- L0 57/64 + L1 124/128 + L2 253/256 都 = Stage 1 utilization (没恶化)
- L3 max = 13 (最多 13 个 items 共享同一 4-digit prefix)

## 3. Stage 3 T5-mini 训练启动

```
PID 3491581, GPU 1, 80% util
Training Epoch 0: 100% (515 batches, 23 sec @ ~22 it/s)
~80 min ETA for 200 epochs
```

**Recipe** (跟 task84 baseline 一致):
- T5-mini 6L encoder + 4L decoder
- d_model=128, d_ff=1024, 6 heads, d_kv=64
- lr=1e-4, batch=256, infer=96, epochs=200
- seed=42 (项目硬约束)
- early_stop=20 (NDCG@20 plateau)
- R12 ckpt save per best NDCG@20

## 4. Stage 4 eval driver 预写

`scripts/task276_stage4_eval.py` 已写, 跟 stage3 evaluate() 函数一致:
- Load HG_Rec_best.pth
- Build test.parquet dataset
- evaluate(model, test_dataloader, topk_list=[5,10,20], beam_size=20)
- Save metrics JSON
- GO/NO-GO check vs HG-Rec baseline 0.1020

## 5. 修过的 3 个 bug

1. ❌ task84_hgrec_stage2_codebook.py 硬编码 → ✅ 写新 ckpt-agnostic driver
2. ❌ `from data.dataset import EmbDataset` → ✅ `from model.utils import EmbDataset`
3. ❌ assignment_mode_list str "shared,shared,shared" (length 20) → ✅ parse 逗号分隔 to List[str]
4. ❌ Stage 3 code_path 是 basename (Stage 3 期望 dataset_name + code_path) → ✅ copy .npy 到 `HG-Rec/dataset/Instruments/Instruments_A2_t5_hrqvae_poincare.npy`

## 6. 物理产物

```
verdicts/task276_a2_stage2_inference_result.md  (本文件)
descriptions/task276_a2_stage2_inference.md
scripts/task276_stage2_inference.py  (dedicated driver, ckpt-agnostic)
scripts/task276_stage2_inference.sh  (launcher)
scripts/task276_stage3_train.sh  (launcher)
scripts/task276_stage4_eval.py  (eval driver, pre-written)
products/task276/stage2/A2_t5_hrqvae_poincare.npy  (Stage 2 产物, 9922x4 int)
HG-Rec/dataset/Instruments/Instruments_A2_t5_hrqvae_poincare.npy  (copy for Stage 3 expected path)
products/task276/stage3/<run_id>/  (Stage 3 ckpt 落盘中)
logs/task276/stage2_inference_*.log
logs/task276/stage3_train_*.log
```

## 7. 当前状态

| 项目 | 值 |
|------|-----|
| Stage 2 | ✅ 完成 |
| Stage 3 | 🟡 训练中 (PID 3491581, GPU 1, ~80 min ETA) |
| Stage 4 | 🔵 待启动 (Stage 3 best ckpt 落盘后) |
| GPU 利用 | GPU 1 80% / 0/2/3 idle |

result: Task #276 — Stage 2 SID inference 完成 (9922 unique, Sinkhorn 30 + 4th-digit dedup). Stage 3 T5-mini 训练启动 (PID 3491581, GPU 1, ~80 min ETA, R12 ckpt save per best NDCG@20). Stage 4 eval driver 预写好等 Stage 3 ckpt. 下个 cron tick: 检查 Stage 3 epoch 进度, 早期 NDCG@20 数字, 决定是否提前 NO-GO.