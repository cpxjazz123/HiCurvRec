# Issue #239 — VERDICT

- issue 正文 sha256：`7a1ee963ef2b0432888b7587b31cd5ac42a603ed3136d98c4b7894065abebfc1`
- 清单：16 条（C1–C16），`RUN_LOG.md`
- 候选：`stage2_RQ-VAE/TIGER_curvloss_rq`（2026-11-23）

## 逐条对账

| 条目 | issue 要求 | 实际值 | 证据 | 判定 |
|---|---|---|---|---|
| C1 | 从冻结基线派生独立候选，复用 #238 机制实现，不依赖 #237 | `diff -r` 仅 5 个机制文件 + `ARM_NAME` | `git show` / diff | **PASS** |
| C2 | 码字选择/残差/欧氏代价/Sinkhorn/退化守卫与 TIGER 逐字节相同 | `model/layers.py`、`distances/__init__.py`、`distances/distance.py`、`config/train_config.json` sha256 与基线一致 | `tests/test_quantiser_unchanged.py`（5 项） | **PASS** |
| C3 | 硬 q 上算 `K_q`；沿用 #238 三元组/参考集/尺度/分层；不可达不填零 | 191,984 三元组、64 参考、正/负/近零 = 171,700/20,151/133 | `calibration.json`、`evaluation.json` | **PASS** |
| C4 | 单一新增损失；混合距离不进 residual→codeword 代价 | 注册表只有 `euclidean`；入口 `DISTANCE is distances.distance.distances` | 同上测试 | **PASS** |
| C5 | 图只用训练行为；监督/标定/评估隔离；方向未建模声明 | `metadata.json` 的 train.parquet 哈希一致；报告声明「方向未建模」 | `INFRA` + 本文 | **PASS** |
| C6 | 训练前冻结 λ/尺度/Huber；不扫参不多种子 | `calibration.json` created 12:59，Stage2 启动 13:0x；λ 由 5% 规则一次决定（5.31e-6） | `calibration.json`、`training_metrics.jsonl` 的 `curv_weight` | **PASS** |
| C7 | 复核 #238 失败根因并给证据 | **三个码本 `embed.weight` 梯度 NaN**（非尺度/非分配）；两条 0/0 已修 | `diagnose_238.py`、`diagnose_238.json`、（已评论） | **PASS** |
| C8 | 欧氏代价与冻结 TIGER 相同 | 公式数值一致（rtol 1e-6） | `test_quantiser_unchanged.py` | **PASS** |
| C9 | STE 路由与梯度正确、无 NaN/Inf、禁用结构项等价 | smoke：`dead_terms=[]`、`nonfinite_terms=[]`、全步长正；曲率项 grad 3.547 | `/tmp/genrec-mcp/smoke_TIGER_curvloss_rq.log` | **PASS** |
| C10 | 单测 + 四卡 smoke + 准入门槛通过后才训练一次 | smoke PASS → Stage2 一次（3000 epoch） | 同上 | **PASS** |
| C11 | 唯一基线为冻结 TIGER；768/32/三层 256/重建不变 | config 逐键相同 | 测试 | **PASS** |
| C12 | `z`/硬 q/SID 重建 q 三路曲率指标 | MAE 2.13→0.89（z）、2.06→0.93（q）、2.68→0.93（SID）；Spearman 0.05→0.73 | `evaluation.json` | **PASS** |
| C13 | SID 结构指标 | 碰撞 0.1693→0.1425；熵 [6.96/6.55/7.95]→[6.96/6.55/7.95]；重建 MSE 9.6e-5→6.8e-5；重导 80 项后一致 | `evaluation.json`、`export_manifest.json` | **PASS** |
| C14 | 硬 q 曲率保真改善 + 报告权衡 + SID 不不可接受退化 | 改善成立；**权衡：负曲率符号一致率 0.485（正 0.970）**；SID 结构未退化（碰撞更低、熵持平） | `evaluation.json` | **PASS（附权衡）** |
| C15 | 通过准入后一次 Stage3；报告 Recall/NDCG、SID 哈希；不称显著 | 见下节 | `test_final.json` | 待填 |
| C16 | 交付实现/配置/测试/smoke 证据/日志/报告/哈希；不覆盖冻结资产；不创建额外 Issue | 全部存在；基线树未改动；issue 数未增 | `git status`、`gh issue list` | **PASS** |

## Stage3（C15）

（Stage3 运行结束后填入：`test_recall@10`、`test_ndcg@10`、SID sha256、与冻结 TIGER 的差值，
并按 issue 要求结合 0.00107 的波动判定，不将单次小幅差值称为显著。）

## 已知框架缺口（记录，不隐瞒）

`stage3_start` 返回 `layer_record.passed=false`：

```
Error while finding module specification for
'tools.research.science.sid_layer_analysis' (ModuleNotFoundError)
```

§63 的分层指标工具随 §94 清理被删除，**该记录当前无法由工具生成**。它不阻断 Stage3（`started: true`）。
本 issue 已用手写脚本产出等价表（`evaluation.json`：碰撞率、各层利用率/熵、前缀共享、重建 MSE、重导一致性），
缺口仅为「工具化」。

## 关键结论（不依赖 Stage3）

1. **#238 的阻塞是梯度问题，不是科学结论**：三个码本梯度 NaN，源于球面块 `sqrt` 在重合点与球映射缺范数下界；
   修复后同一诊断三步全绿。这条同时解释了「为什么混合距离进 Sinkhorn 会炸」而「进 loss 不会」。
2. **曲率监督在硬 q 与 SID 重建 q 上确实生效**：MAE 降 54%/65%，Spearman 从 ~0 到 0.73。
3. **但它主要是「正曲率拟合」**：负曲率组符号一致率仅 0.485，接近随机。这是需要后续机制解决的明确短板。
4. **SID 结构未退化**：碰撞更低（0.1425 vs 0.1693）、熵持平、重建 MSE 更低、行为边关联更强（0.66 vs 0.54）。
