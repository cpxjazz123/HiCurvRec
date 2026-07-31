# Task #428 / Issue #136 [方向C Gate4] 重建真实 history→SID 评估协议后复跑 dual-gate

## 目标

Issue #133 用 proxy SID[:3] dataset 跑 R@K=0 (protocol mismatch)。Issue #136 重建 Musical_Instruments 真实 Stage4 评估协议: GenRecDataset(test.parquet) + code_path (SID npy) + 真实 history→4-token/item sequences → 跑 control + adapter 双次, 验证 #129 dual-gate ckpt 在真实 Stage 4 评估下是否 PASS baseline。

## 实施

- 脚本: `scripts/task428_issue136_stage4_rebuild.py`
- WrappedHGRec 类: HG_Rec + 可选 DualGateAdapter (issue #129) on input embedding via inputs_embeds hook
- 复用 task84_hgrec_stage3_train.evaluate (Stage 4 eval 函数)
- 复用 GenRecDataset + GenRecDataLoader (real dataset protocol, mode='evaluation')
- T5 ckpt + SID npy + adapter ckpt 三件套 SHA256 校验 (R139 reproducibility triangle)

## 预期产物

- `products/task428_issue136_stage4_rebuild/verdict.json`
- `verdicts/task428_issue136_gate4_*.md`
- Control + adapter 双次 eval, 6 metrics vs HG-Rec baseline

## Gate 4 决策

- PASS: 双次 eval 均有效 AND adapter R@10 > 0.1020 (baseline)
- FAIL: 双次任一 invalid 或 adapter R@10 ≤ 0.1020

## 关联

- Issue #136: rebuild real Stage 4 dataset + rerun #129 dual-gate ckpt
- 联立 #133 (proxy R@K=0) → #136 (real protocol rebuild) 是 protocol mismatch 解药
- R7: 启动前 nvidia-smi 选空闲 GPU (cuda:2, 4×L40S 全空闲)
- R17 + R20: commit + verdict 必须详细 4 Gate 回答
- R139 reproducibility triangle: SHA256 ckpt + SID + eval script 三件套必须落盘