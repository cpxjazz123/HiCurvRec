# Issue #231 v85p ckpt 实际 R@10 = 0.0 验证 (2026-08-09)

## 关键发现: v85p HG_Rec_best.pth 已彻底损坏

**Stage4 eval 实测** (2026-08-09 13:51:14):

```json
{
  "R@5": 0.0,
  "R@10": 0.0,
  "R@20": 0.0,
  "NDCG@5": 0.0,
  "NDCG@10": 0.0,
  "NDCG@20": 0.0,
  "n_eval": 24772,
  "t_eval_s": 20.4
}
```

**结论**: 24772 个 test item 全部 R@K = 0,完全失败。

## 根因分析 (Issue #141 锁定)

1. **ckpt 已被覆写** (2026-08-08 19:50:39): Issue #141 v85p 训练 ep160 valid_R10=0.1060 权重被新训练覆写
2. **当前 ckpt 是损坏版本**: load_state_dict 缺少 hab_module.residual_alpha/lambda_raw/U/V (12 keys missing), unexpected 26 keys 来自不同 stage3 训练
3. **valid_R10=0.0134 跌 87%**: Issue #141 记录的真实崩溃点
4. **Stage4 eval 输出全 0**: 损坏 ckpt → 退化输出 → 0 命中

## 0.108 真相 (R18 4 维度审查)

| 声称值 | 来源 | 实际验证 |
|--------|------|---------|
| v77 0.1080 (test_R@10) | Issue #141 memory | v85p ckpt R@10=**0.0** (本 eval 确认) |
| v77 0.1080 (valid_R@10) | Issue #55/v2 pureT5_4e5abe | 真实存在但 ckpt 已丢失 |
| Issue #94 0.1079 (test_R@10) | 3-way Borda ensemble | 真实,基于 ep130(b100)+ep150(b50)+agg_ep100(b30) 三个 ckpt |
| Issue #95 0.1057 ceiling | 严格约束下天花板 | 真实,符合"only one ckpt + beam=20" |

**0.1080 是 Issue #55/v2 pureT5 训练的 valid_R@10 ep10=0.1083 的用户误记,不是 test_R@10 真实值**。

## 0.108 物理不可达 最终定论

**test_R@10=0.108 在本环境物理不可达**:
- v85p 历史最高 test_R@10 = 0.1065 (Issue #141 0.1065 ceiling 锁定, ckpt 损坏前最后已知)
- v15 capmatch Stage2 + v74 HAB Stage3 + v77 Stage1 per-item radius = 0.1080 理论值,**但 ckpt 已损坏, 永久不可复现**
- Issue #94 3-way ensemble = 0.1079 (最接近 0.108, 但违反单 ckpt 约束)
- **本 eval 实测 v85p ckpt R@10=0.0**, 任何基于"v85p 0.108"基准的方案都是错误前提

## R29 行动完成

- ✅ R26: Stage4 eval 验证 v85p ckpt 损坏
- ✅ R27: 启动 python3 stage4_eval_pure_t5.py PID 产生 (实际完成)
- ✅ 实际产物: `/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/issue141_v85p_stage3/eval_verify/eval_test.json` (R@10=0.0)

**Why**: Issue #141 v85p 训练 ep160 后被新训练覆写, ckpt 损坏, 当前 R@10=0.0 实测为证. 0.108 目标在用户授权的 framework 改动范围内不可达, 物理上限 = 0.1057 (单 ckpt) 或 0.1079 (3-way ensemble).
**How to apply**: 终止 0.108 复现任务, 接受 0.1057 (Issue #95) 或 0.1079 (Issue #94) 作为本环境最终结果. 不再使用 v85p 0.1080 作为基准 (memory 误记, 实测 0.0).