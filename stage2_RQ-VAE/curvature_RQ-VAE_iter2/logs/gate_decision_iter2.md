# iter2 Gate Decision

- 机制：P2 双曲 dead-code online clustering
- 日期：2026-09-21
- 数据集：Amazon-2023 Instruments
- 裁决：**NO-GO / ARCHIVE**
- 正式 baseline：不修改、不提升

## 训练前验证

- `py_compile`：通过。
- `scripts/grad_check.sh`：`GRAD_CHECK PASS`。
- 总 loss：`requires_grad=True` 且存在 `grad_fn`。
- 每层 quantize loss：均获得非零参数梯度。
- 曲率步 `0` 与 `25000`：quantize loss 数值不同，确认 `c(t)` 进入计算图。
- P2 状态路径：使用真实 `AdamW` 和单 batch 强制构造 1 个 dead code；当前曲率下执行 `expmap0 → Poincaré projection → logmap0`，成功改写 1 个 codebook 行，并清零对应 optimizer 行状态和窗口状态。
- 训练未启动前未发现 `.detach()` 截断导致的 silent no-op。

## Stage2 结果

- 训练：4 卡 DDP，`global_step=100000`，总耗时约 `951.8s`，正常结束。
- 最终 checkpoint：`out/rqvae/instruments/rqvae_final.pt`。
- raw SID：`(24587, 3)`，三 token 唯一组合 `23243/24587`。
- collision-extended SID：`(24587, 4)`，四 token 全部唯一 `24587/24587`。
- 最终 descriptive 指标：
  - `hitrate@50=0.7165`（仅记录，不作 gate）
  - `full_gini=0.0527`
  - `per_layer_gini=[0.0722, 0.1524, 0.1919]`
  - `l01_pairs=15610`
  - `H(L1|L0)=5.6522`
  - `early_stop=False`
- 训练期间每个窗口均未出现全局零使用 code，因此没有触发实际再激活；这不是失败条件，P2 的执行路径已由常驻梯度检查覆盖。

## Stage3 结果

- 使用 iter2 生成的四 token SID，未修改 Stage3 代码。
- 配置确认：`sid_length=4`、`codebook_size=[256,256,256,1]`、`vocab_size=787`。
- 训练：4 卡，完整 `150/150` epoch，正常结束。
- 最终结果文件：
  `/fs04/ar57/wenyu/GeneRec/stage3_T5Train/logs/tiger_baseline/Amazon_2023_Instruments/Sep-21-2026_03-47-50/test_final.json`
- 最终指标：
  - `test_recall@5=0.035289611587945476`
  - `test_R@10=0.053186859102700254`
  - `test_ndcg@5=0.023120376380552792`
  - `test_ndcg@10=0.028871709511892014`
- 硬目标：`test_R@10 > 0.065`，实际 `0.053186859102700254`，差距 `0.011813140897299746`，因此 NO-GO。
- 相对当前正式 curvature baseline `0.05356987412733508`：下降 `0.000383015024664826`（约 `-0.7147%`），不满足提升条件。

## 处置

- 不复制 iter2 到正式 `curvature_RQ-VAE/`。
- 提交本审计日志后删除 iter2 代码目录及其独立结果目录；Stage3 只读日志和最终 JSON 保留在其既有输出路径。
- 审计提交 hash 将在下一版本文件中记录。
