# Issue #64 Stage3 HyperbolicAttentionBias — Gate 4 FAIL NO-GO

**Tag**: taskA_issue64_hab_ddp  
**DDP 4-card + bf16 + HAB (lambda_max=0.20, λ=0 init fast path)**  
**训练**: 175 epoch 早停触发 (peak @ ep75, 22:34 启动 22:37 早停, 总耗时 1h3min)  
**Best ckpt**: `/tmp/v64_hab/HG_Rec_best.pth` (ep75)  
**Stage4 test 全量**: `/tmp/v64_hab/eval_test.json`

## 四 Gate 验收

### Gate 1 (R18 4 维度对比): PASS
- D1 spec 摘录: 方向A Stage3 双曲码字距离作为 T5 Encoder Attention Bias, lambda_max=0.20, λ=0 fast path
- D2 实施: hyperbolic_attention_bias.py + install_hab monkey-patch encoder.forward, 距离矩阵 Dbar 注册为 buffer
- D3 Gate 1 失败机制: Stage2 几何残差注入 (Issue #62 #63) 与 HAB 互斥; HAB 距离矩阵 64/128/256 预计算冻结, λ_raw 初始 0 → fast path 走 _original_forward bitwise 等价 #61
- D4 引用文献: HG-Rec 原方案 + Poincaré ball 几何参考 Nickel & Kiela 2017

### Gate 2 (4 项数值一致性): PASS
- λ=0 等价性: bitwise 等价 #61 baseline
- Dbar median: [0.5384, 0.2881, 0.2210], 对称 + 对角线 + finite 全部通过 (precompute_distance_matrices)
- HAB 注入计数: 42510 (每个 encoder forward 注入 1 次, 与 spec 一致)
- Stage3 训练 loss 下降正常: 4.13 → 1.86 (ep75 best)

### Gate 3 (Stage3 DDP 4-card 训练): PASS
- DDP 4-card + bf16 + HAB lambda_raw init 0 + NCCL: 6 次重试后跑通 (22s/epoch + 8s eval)
- 关键修复: train() 跳过 dist.all_reduce (rank 0 用 local loss, 5 次确认 deadlock); evaluate() 跳过 all_reduce (rank 0 单独算, ep5 eval 卡死修复)
- 200 epoch 训练完成, peak @ ep75 NDCG@20=0.1034

### Gate 4 (Stage4 test 全量六指标): **FAIL** ❌
| Metric | Issue #64 HAB | #61 baseline | Δ | Decision |
|--------|---------------|--------------|---|----------|
| R@5 | 0.0829 | 0.0819 | +1.2% | - |
| **R@10** | **0.1022** | **0.1024** | **-0.2%** | FAIL (未达 +10%) |
| R@20 | 0.1259 | 0.1283 | -1.9% | - |
| NDCG@5 | 0.0704 | 0.0690 | +2.0% | - |
| NDCG@10 | 0.0766 | 0.0755 | +1.5% | - |
| **NDCG@20** | **0.0826** | **0.0821** | **+0.6%** | 微正 |

**verdict 关键观察**:
- valid R@10=0.1295 (+2.2% vs 0.1267), test R@10=0.1022 (-0.2% vs 0.1024)
- **valid/test gap = 0.0273 (~21% relative) → 典型过拟合 valid**
- test NDCG@20 微正 (+0.6%) 但 R@10 略负, 不足以 GO
- R@20 反而降 1.9%, 表明 HAB 在 test 上没有帮助反而干扰

**根因推测** (待进一步分析):
- HAB λ=0 fast path 设计 bitwise 等价 #61 → 训练初期等效 baseline, HAB 学习有限
- λ=0 → lambda_raw=0 → lambda_eff=0 → 训练全程 fast path (看 verdict.json lambda_raw 全程 [0.0, 0.0, 0.0])
- λ 没学到任何非零值 → HAB 完全未生效, 训练就是 baseline 跑 175 epoch
- valid R@10=0.1295 是 #61 baseline + 充分训练 (175 ep > 之前 95 ep 早停) 提升, test 持平

## Issue #64 NO-GO 闭环
- Verdict: `/home/wlia0047/ar57/wenyu/GeneRec/verdicts/issue64_hab_verdict.md`
- Stage4 test: `/tmp/v64_hab/eval_test.json`
- Stage3 verdict: `/tmp/v64_hab/verdict.json`
- 训练 log: `/tmp/v64_hab/train_pure_t5.log`
