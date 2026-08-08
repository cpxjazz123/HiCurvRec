# Issue #32 0.108 不可复现 (资源丢失)

**Date**: 2026-08-09
**Status**: ❌ NO-GO (资源丢失)

## 调查

复现 valid_R@10=0.1083 (pureT5_4e5abe ep10) 需要:
1. **Stage2 v3e poincare SID** (`taskA_stage2_v3e_poincare_mix1x/sid_output.npy`, sha=5c058531)
2. **pureT5_4e5abe 训练 ckpt** (`taskA_stage3_pureT5_4e5abe/HG_Rec_best.pth`)

**两者都已从磁盘清理**:
- `_history/` 下只剩 3 个 Stage2 目录 (hyp_v2, issue61, v15), v3e_poincare_mix1x 已删除
- `taskA_stage3_pureT5_4e5abe/` 目录不存在, HG_Rec_best.pth 已丢失

## 仅存的证据

- `/fs04/ar57/wenyu/GeneRec/taskA/_logs/pureT5_taskA_run.log` (完整训练 log, 37 epoch, 含 valid_R@10 轨迹)
- `/home/wlia0047/.claude/projects/-fs04-ar57-wenyu-GeneRec/memory/issue55-v2-pureT5-0.108-valid.md` (memory 修正记录)

## 4 Gate 答复

### Gate 1 (Spec): 复现目标 = pureT5_4e5abe ep10 valid_R@10=0.1083 (Issue #55/v2)
### Gate 2 (实施): 查 _history/_logs, SID + ckpt 均丢失, 无法 Stage4 eval / 续训
### Gate 3 (Gate 1 失败机制): 资源清理不可逆, 唯一复现路径 = **从 Stage2 v3e poincare 重新生成 SID + 重新训练 pureT5_4e5abe 配置** (但这违反 R31 单脚本原则 — 主脚本硬编码 LR=4e-4 batch=1024, 不支持 lr=1e-4 batch=256)
### Gate 4 (FAIL): **0.108 不可复现** (资源丢失), 实际本环境 SOTA test_R@10 = v4 0.1031 (已超 baseline +0.0017)

## 结论

1. **0.108 (valid_R@10=0.1083) 不可复现**: SID + ckpt 都从磁盘清理
2. **本环境 test_R@10 SOTA = v4 0.1031** (已超 baseline 0.1024)
3. **4 组 SID 探索全部 NO-GO**: hyp_v2 0.1031 > v15 0.0966 > equal128 0.0942 > equal256 0.0835
4. **用户应接受 v4 0.1031 为新基线**

## 复现 pureT5_4e5abe 唯一可行路径 (需用户批准 R31 例外)

1. **重新训练 Stage2 v3e poincare_mix1x** (用 taskA_stage2.py wrapper + poincare_mix1x 配置)
2. **生成 SID sha=5c058531...**
3. **重新训练 Stage3 pureT5_4e5abe** (单卡 batch=256 lr=1e-4, 但主脚本硬编码 LR=4e-4 batch=1024 → 需 fork Stage3 主脚本)
4. **Stage4 eval** 拿 test_R@10 (期望 ≈ 0.1015)

但这违反 R31 单脚本原则, **需用户明确批准 R31 例外**.
