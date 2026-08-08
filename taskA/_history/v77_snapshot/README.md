# v77 Snapshot — 4 阶段脚本固化版

日期: 2026-08-08
来源 commit: 273899b40cc94074d376e51e694b1a1f37d2f5ea (main HEAD)
对应 SID: 06af0fedf4907b05b2e1efd45ab6a96dd4aa36fa14db05b2c8f285283b7f1944 (hyp_v2 Stage2 SID)
对应 SHA: test_R10=**0.1080** (整体 SOTA), valid=0.1312, ratio=1.215

---

## v77 4 阶段流水线

| 阶段 | 脚本 | 配置参数 |
|---|---|---|
| **Stage 1** | `stage1_hyperbolic.py` | `--tag hyp_v2 --r_max 0.99 --r_mode heuristic` (per-item radius) |
| **Stage 2** | `taskA_stage2.py` | `ITEM_EMB_NPY=taskA/_history/taskA_stage1_hyp_v2/item_emb_u32.npy` (Stage1 hyp_v2 输出) |
| **Stage 3** | `stage3_train_pure_t5.py` | `--hyperbolic_attn_bias --enable_residual_hab --hab_lambda_max 0.20 --residual_alpha_init -20.0`, `num_decoder_layers=4`, LR=4e-4 const (v85 之前) |
| **Stage 4** | `stage4_eval_pure_t5.py` | 与 Stage3 同 `--hyperbolic_attn_bias` |

---

## 启动命令示例 (DDP 4 卡)

### Stage 1
```bash
cd /fs04/ar57/wenyu/GeneRec
CUDA_VISIBLE_DEVICES=0 python3 -m common.stage1.stage1_hyperbolic --tag hyp_v2 --r_max 0.99 --r_mode heuristic
```

### Stage 2 (产物 → `taskA/_history/taskA_stage2_hyp_v2_capmatch_1000ep/`)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.run \
  --nproc_per_node=4 --master_port=29500 --standalone \
  common/stage1/../taskA/stage2/taskA_stage2.py --no_mlr 0 --n_epochs 1000
```

### Stage 3
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 /home/wlia0047/ar57_scratch/wenyu/genrec_env/lib/python3.10/site-packages/torch/distributed/run.py \
  --nproc_per_node=4 --master_port=29500 --standalone \
  /fs04/ar57/wenyu/GeneRec/common/stage3/stage3_train_pure_t5.py \
  --sid_npy /home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_hyp_v2_capmatch_1000ep/sid_output.npy \
  --expected_sid_sha 06af0fedf4907b05b2e1efd45ab6a96dd4aa36fa14db05b2c8f285283b7f1944 \
  --product_dir /path/to/output \
  --tag v77 \
  --hyperbolic_attn_bias --enable_residual_hab \
  --hab_lambda_max 0.20 --residual_alpha_init -20.0 \
  --num_decoder_layers 4 --lr 4e-4
```

### Stage 4
```bash
CUDA_VISIBLE_DEVICES=0 python3 common/stage4/stage4_eval_pure_t5.py \
  --ckpt /path/to/HG_Rec_best.pth \
  --sid_npy /home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_hyp_v2_capmatch_1000ep/sid_output.npy \
  --expected_sid_sha 06af0fedf4907b05b2e1efd45ab6a96dd4aa36fa14db05b2c8f285283b7f1944 \
  --hyperbolic_attn_bias --enable_residual_hab \
  --hab_lambda_max 0.20 --residual_alpha_init -20.0
```

---

## 与 baseline 差异

| 维度 | v77 | baseline |
|---|---|---|
| Stage 1 TAG | `hyp_v2` (per-item radius, R_MAX=0.99) | `baseline` (无 per-item radius, 标准 sentence-t5 输出) |
| Stage 2 SID | hyp_v2 (sha=06af0fed) | v15 (sha=5f8331cc) |
| Stage 3 num_decoder_layers | 4 | 6 (v85p) |
| Stage 3 LR | const 4e-4 | cosine 1e-3, warmup 10% (v85p) |
| HAB λ_max | 0.20 | 0.20 |
| residual_alpha_init | -20.0 | 0.5 (v85p) |

---

## 文件清单

- `stage1_hyperbolic.py` — common/stage1/stage1_hyperbolic.py 副本
- `taskA_stage2.py` — taskA/stage2/taskA_stage2.py 副本
- `stage3_train_pure_t5.py` — common/stage3/stage3_train_pure_t5.py 副本
- `stage4_eval_pure_t5.py` — common/stage4/stage4_eval_pure_t5.py 副本
- `hyperbolic_attention_bias.py` — common/hyperbolic_attention_bias.py 副本

---

## 备注

v77 的 Stage3 当前主脚本 (含 Branch Curvature 改动) 已恢复到 v85p 干净版 (commit ea0fc9b), 当前 main HEAD 273899b = 干净 baseline。
Branch Curvature 改动保留在 `common/stage3/stage3_train_pure_t5.py` 历史 commit 中 (c15d038 + e6b2752),可通过 git checkout 还原。