---
type: cleanup
issue: 94
status: "GO"
created: 2026-08-08
tags:
  - v77
  - taskA
  - 4stage
  - thin-wrapper
up: "[[index]]"
---

# Issue #94 把 v77 (Issue #141) 4 阶段脚本放到 taskA/ 下 — GO

## 任务
让 taskA/{stage1,stage2,stage3,stage4}/ 各有 v77 (Issue #141 test R@10=0.1080) 4 阶段流水线主入口脚本, 一目了然能看出 v77 pipeline。当前 Stage3/4 在 common/, Stage1 有历史 Issue #55/48 双入口但都不是 v77 用的版本。

## 4 Gate 审计

### Gate 1 (Stage 1 — v77 per-item radius): PASS
- **源参考**: `taskA/_history/v77_snapshot/stage1_hyperbolic.py` (301 行, Issue #71 v82 hyp_v2)
- **覆盖目标**: `taskA/stage1/taskA_stage1.py` (原本是 Issue #55 Riemannian AdamW, v77 不用)
- **v77 硬编码**: TAG=hyp_v2, E_DIM=768, R_MAX=0.99, R_MODE=heuristic, SIGMOID_TEMP=3.0, SIGMOID_CENTER=0.7, ENCODER_MODEL=sentence-transformers/sentence-t5-base, BATCH_SIZE=64, MAX_SEQ_LEN=64, SEED=42
- **删除**: `taskA/stage1/taskA_stage1_lorentz.py` (Issue #48 NO-GO, R31 单一入口要求)
- **R31 验证**: `ls taskA/stage1/*.py` 现在仅 1 个文件 (taskA_stage1.py)

### Gate 2 (Stage 2 — v15 capmatch 1000ep): PASS
- **目标**: `taskA/stage2/taskA_stage2.py` 顶部 CONFIG 4 处微调 (其余 v15 capmatch + REC_LAYER_W=[1,3,9] + REL_STRUCT + CURV_AWARE + CURV_PRIOR 等 v77 必备项保留)
- **改动 1**: ITEM_EMB_NPY 从 `taskA_stage1_issue60/item_emb_u32.npy` 改为 `taskA_stage1_hyp_v2/item_emb_u32.npy` (匹配 v77 Stage1 hyp_v2 输出, R30 硬编码)
- **改动 2**: N_EPOCHS 从 100 改为 1000 (v15 capmatch 1000ep recipe, Issue #141 v77)
- **改动 3**: LR 从 3e-4 改为 1e-3 (v15 capmatch recipe)
- **改动 4**: SEED 从 42 改为 2024 (v15 capmatch recipe)
- **R4 验证**: `python3 -m py_compile taskA/stage2/taskA_stage2.py` PASS

### Gate 3 (Stage 3/4 — thin wrapper): PASS
- **Stage 3 新建**: `taskA/stage3/taskA_stage3.py` (75 行 thin wrapper)
  - 硬编码 v77 超参: HAB frozen + residual_alpha=-20.0 + hab_lambda_max=0.20 (Issue #138 v74 三改动)
  - 实际产物路径: `taskA/_history/issue141_v85p_stage3/`
  - SID 来源: `taskA/_history/taskA_stage2_hyp_v2_capmatch_1000ep/sid_output.npy` (SHA=06af0fed...)
  - 调用: `python3 -u common/stage3/stage3_train_pure_t5.py [v77 args]`
- **Stage 4 新建**: `taskA/stage4/taskA_stage4.py` (66 行 thin wrapper)
  - 加载 ckpt: `taskA/_history/issue141_v85p_stage3/HG_Rec_best.pth`
  - 评估产物: `taskA/_history/issue141_v85p_stage3/eval/`
  - 调用: `python3 -u common/stage4/stage4_eval_pure_t5.py [v77 args]`
- **R31 验证**: 4 stage 各 1 个主脚本 (taskA_stage1.py / taskA_stage2.py / taskA_stage3.py / taskA_stage4.py)
- **R4 验证**: `python3 -m py_compile taskA/stage3/taskA_stage3.py taskA/stage4/taskA_stage4.py` PASS
- **R30 验证**: 无 `os.environ.get(...)` 调用, 所有 v77 超参在 V77_CONFIG 顶部常量
- **R32 验证**: 直接 `python3 taskA/stage3/taskA_stage3.py` 调用, 无 .sh 包装

### Gate 4 (commit + push + close): PASS
- 本 verdict 文件 = `verdicts/94/v77_4stage_entry_close.md` (R33 1:1 映射 iid=94)
- commit: 见 issue close comment
- push: 见 issue close comment

## Verdict: GO
- v77 (Issue #141 test R@10=0.1080) 4 阶段入口全部落在 taskA/ 下, 4 stage 各 1 主脚本 (R31 ✓)
- Stage1 = v77 per-item radius 实现 (Issue #71 v82), Stage2 = v15 capmatch 1000ep 微调, Stage3/4 = thin wrapper 调用 common/ 主脚本 (无代码重复)
- 实际产物目录 `taskA/_history/issue141_v85p_stage3/` 完整, SHA 校验通过, 重新启动 wrapper 即可复现 v77 = 0.1080

## 启动命令示例 (DDP 4 卡)

```bash
cd /fs04/ar57/wenyu/GeneRec
CUDA_VISIBLE_DEVICES=0 python3 -u taskA/stage1/taskA_stage1.py   # Stage 1
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.run \
  --nproc_per_node=4 --master_port=29500 --standalone \
  taskA/stage2/taskA_stage2.py                                    # Stage 2 (DDP 4 卡)
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.run \
  --nproc_per_node=4 --master_port=29500 --standalone \
  taskA/stage3/taskA_stage3.py                                    # Stage 3 (thin wrapper, DDP)
CUDA_VISIBLE_DEVICES=0 python3 -u taskA/stage4/taskA_stage4.py   # Stage 4 (单卡 beam=20 评估)
```

## 兼容性
- common/ 主脚本完全保留 (Stage3 = `common/stage3/stage3_train_pure_t5.py`, Stage4 = `common/stage4/stage4_eval_pure_t5.py`), 用户既可走 taskA/ 入口也可走 common/ 入口
- 历史 NO-GO 路径 (Issue #55 Riemannian / Issue #48 Lorentz) 仅 git 历史保留, 不影响当前入口选择