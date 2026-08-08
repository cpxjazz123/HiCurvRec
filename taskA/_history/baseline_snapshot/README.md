# baseline Snapshot — 4 阶段脚本固化版 (HG-Rec Task #84 baseline)

日期: 2026-08-08
来源 commit: 273899b40cc94074d376e51e694b1a1f37d2f5ea (main HEAD, 恢复 v85p 干净版)
对应 SID: 5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07 (v15 Stage2 SID)
基线 Task #84: valid_R10=0.1267, test_R10=**0.1024**

---

## baseline 4 阶段流水线 (与 HG-Rec 一致)

| 阶段 | 脚本 | 配置参数 |
|---|---|---|
| **Stage 1** | `stage1_hyperbolic.py` | `--tag baseline` (无 per-item radius, sentence-t5-base 标准 Euclidean) |
| **Stage 2** | `taskA_stage2.py` | `ITEM_EMB_NPY=taskA/_history/taskA_stage1_hyp_v2/../baseline/item_emb_u32.npy` (基线 Stage1 输出) |
| **Stage 3** | `stage3_train_pure_t5.py` | `--hyperbolic_attn_bias --enable_residual_hab --hab_lambda_max 0.20 --residual_alpha_init 0.5`, `num_decoder_layers=6`, LR=1e-3 cosine 10%/0.05 (v85p baseline) |
| **Stage 4** | `stage4_eval_pure_t5.py` | 与 Stage3 同 `--hyperbolic_attn_bias` |

---

## 启动命令示例 (DDP 4 卡)

### Stage 1
```bash
cd /fs04/ar57/wenyu/GeneRec
CUDA_VISIBLE_DEVICES=0 python3 -m common.stage1.stage1_hyperbolic --tag baseline
```

### Stage 2
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.run \
  --nproc_per_node=4 --master_port=29500 --standalone \
  taskA/stage2/taskA_stage2.py --no_mlr 0 --n_epochs 1000
```

### Stage 3 (复现 HG-Rec baseline Task #84)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 /home/wlia0047/ar57_scratch/wenyu/genrec_env/lib/python3.10/site-packages/torch/distributed/run.py \
  --nproc_per_node=4 --master_port=29500 --standalone \
  /fs04/ar57/wenyu/GeneRec/common/stage3/stage3_train_pure_t5.py \
  --sid_npy /home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy \
  --expected_sid_sha 5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07 \
  --product_dir /path/to/output \
  --tag baseline \
  --hyperbolic_attn_bias --enable_residual_hab \
  --hab_lambda_max 0.20 --residual_alpha_init 0.5
```

### Stage 4
```bash
CUDA_VISIBLE_DEVICES=0 python3 common/stage4/stage4_eval_pure_t5.py \
  --ckpt /path/to/HG_Rec_best.pth \
  --sid_npy /home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy \
  --expected_sid_sha 5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07 \
  --hyperbolic_attn_bias --enable_residual_hab \
  --hab_lambda_max 0.20 --residual_alpha_init 0.5
```

---

## 文件清单

- `stage1_hyperbolic.py` — common/stage1/stage1_hyperbolic.py 副本
- `taskA_stage2.py` — taskA/stage2/taskA_stage2.py 副本
- `stage3_train_pure_t5.py` — common/stage3/stage3_train_pure_t5.py 副本 (v85p 干净版)
- `stage4_eval_pure_t5.py` — common/stage4/stage4_eval_pure_t5.py 副本
- `hyperbolic_attention_bias.py` — common/hyperbolic_attention_bias.py 副本

---

## 与 v85p SOTA 关系

当前 main HEAD = baseline snapshot = v85p baseline 配置(200+ epoch 后训练产物 test=0.1060)。
`v85p SOTA` 比 baseline (Task #84, test=0.1024) +0.0036 (= +3.5%),比 v77 SOTA -0.0020 (= -1.85%)。

如需还原任何历史实验变体(带 Branch Curvature 等),通过 git 历史 commit 还原:
```bash
git checkout c15d038 -- common/stage3/stage3_train_pure_t5.py  # 含 Branch Curvature
```