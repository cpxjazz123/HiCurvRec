# Issue #240 运行日志

- issue: `cpxjazz123/HiCurvRec#240`
- 标题: Stage2 诊断：完整记录 TIGER_BEHAVIOR_SAMPLE_RQ-VAE 的梯度、裁剪及参数更新轨迹
- 正文快照: `ISSUE_BODY.md`，sha256 `1c0798dc083e400ded1f2a4c01b9562b88b9a7cbaf84a1691553de1a10c42539`
- 冻结时间: 2026-11-23

## 是否需要训练

**需要**。依据 issue「固定协议」第 3 条：「只运行一次带观测的原配方 Stage2，属于诊断复跑」。
验收指标是**逐 optimizer step 的梯度/裁剪/参数更新轨迹**，只能由新权重与新 backward 产生，
不能由任何已有产物复算。issue「完整轨迹与交付」还要求逐 step JSONL 与 epoch 汇总。

同时 issue 明确「本轮不运行 Stage3」，所以只跑 Stage2 一次。

## 冻结验收清单

|编号|判据|阈值|数据|口径|
|---|---|---|---|---|
|C1|核对历史成功版本源码/数据/配置/路径/checkpoint/SID/历史结果哈希|全部记录到 `INFRA.md`|`results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample/`、`results/stage3_T5Train/TIGER_BEHAVIOR_SAMPLE_RQ-VAE__resample/`|sha256 落盘|
|C2|冻结超参不变：768 输入、32 瓶颈、三层 256、3000 epoch、seed 42、**原 batch/DDP**、AdamW、lr、重建/量化损失、行为采样、温度、Sinkhorn、STE、裁剪 1.0；并核实实际每卡/全局 batch|逐键与历史 `train_start` 记录相同|`logs/resample/training_metrics.jsonl` 首行|逐键 diff|
|C3|行为权重 ≤1500 为 0、线性升到 3000 的 0.2；核对三层行为项与梯度路由|`behavior_weight(e)` 复算一致|`train_rqvae.py::behavior_weight`|源码 + 数值复算|
|C4|只运行一次带观测的原配方 Stage2；不重训 TIGER、不扫参、不多种子、不改损失；独立输出目录；不跑 Stage3|1 次|新臂目录 + 新结果目录|run 记录|
|C5|诊断无副作用：不改 RNG/采样/累计/DDP 同步/优化器状态/更新顺序，不引入额外反向传播副作用；小批量 + 四卡 smoke 比较开关前后同一步 loss/grad/param/SID|逐位等价（loss/梯度/参数/SID）|`results/research/issue240/`|smoke 对照|
|C6|全程每步（非抽样）记录 epoch、global step、rank/同步口径、lr、行为权重、batch、各 loss|每 step 一行|JSONL|计数核对 = 步数|
|C7|loss 拆分：重建、commitment、codebook/VQ 逐层与总计、原始完整 RQ-VAE、各层未加权行为项及合计、加权行为项、最终总 loss|与实现系数一致|同上|数值复算 `sum` 一致|
|C8|Encoder/Decoder/L1/L2/L3/全模型的未加权与实际加权梯度 L2/RMS/max abs/非零比例；None 与真零区分；两类比例；分母近零标注|全部字段|同上|逐字段存在性 + 分母守卫|
|C9|梯度方向：重建 vs 行为、原始完整 vs 行为、VQ/commitment vs 行为、三层两两、原始 vs 合成；余弦与点积；零范数 NA；负余弦冲突率与强度|全部对|同上|数值复算|
|C10|裁剪前范数、裁剪系数、是否触发 1.0、裁剪后范数、各模块前后范数；加权梯度之和 = 实际 backward 梯度（DDP 全局同步与误差）|相对误差记录|同上|向量和校验|
|C11|AdamW 前后各模块参数变化 L2/RMS/max/相对率；区分梯度与位移；码字更新与间距；固定探针 z/残差/硬 q 位移与分配变化率|全部字段|同上|逐字段|
|C12|有限性、爆炸/消失、码字/分配退化、异常 batch 与错误原文|无屏蔽|同上|NaN/Inf 标记|
|C13|关键节点单列：首步/50/500/1000/1499/1500/1501/1875/2250/2625/3000；每 epoch 汇总均值/中位数/分位数/极值/裁剪率/冲突率；禁用阶段不混反事实|全部节点存在|`key_steps.md` + `epoch_summary.csv`|逐节点核对|
|C14|逐 step JSONL/Parquet、epoch CSV、曲线、训练日志、配置与哈希、REPORT.md；标明存储与额外开销|全部存在|`results/research/issue240/`|文件清单|
|C15|重建 MSE、各层码本利用率/熵、码字间距、SID 碰撞、行为关系保留；与冻结 TIGER/历史 resample 同口径对比并说明是否重现；历史 Recall 只引用|同口径|同上|数值对比|
|C16|回答六个问题（倍率、何时超过、冲突在哪层/阶段、超 1 频率、裁剪后更新、码本与行为保真随时间），并区分实测/假设；给调度参考不预设最优|全部回答|`REPORT.md`|逐问核对|
|C17|不改现有训练机制、不创建额外 Issue、不训练其他模型；不跑 Stage3|diff 仅在诊断副本|git diff|核对|
|C18|若无法保证无副作用或原配方无法复现，报告阻塞|—|—|—|

## 冻结配方（已核实，与 issue 一致）

|项|历史值|来源|
|---|---|---|
|`EPOCHS`|3000|`train_start.epochs`|
|`BATCH_SIZE_PER_RANK`|**4096**|`train_start.batch_size_per_rank`|
|`global_item_batch`|**4096**|`train_start.global_item_batch`|
|`global_pair_batch`|2048|`train_start.global_pair_batch`|
|`world_size`|**1**（`mode=ddp1`, `cuda_visible_devices="1"`）|`train_start.world_size`|
|`SEED`|42|同上|
|`LR` / `WEIGHT_DECAY`|1e-3 / 1e-4|同上|
|`HIDDEN_SIZES`|(512,256,128)|同上|
|`CODEBOOK`|3 × 256, dim 32|同上|
|`BEHAVIOR_WEIGHT_MAX` / `RAMP_START_EPOCH` / `TEMPERATURE`|0.2 / 1500 / 0.07|源码 433–441|
|`GRADIENT_CLIP_NORM`|1.0|源码 425|
|`train_items`|24,556|`train_start`|

**batch 注释与常量不一致（issue 点名要求核实）**：源码第 410 行注释写「总 batch = 1024 * 4 = 4096」，
但常量为 `BATCH_SIZE_PER_RANK = 4096`，且历史运行记录的 `world_size=1`，因此实际每卡 4096、全局 4096。
本 issue 按**历史实际值**复跑，不改动。

**与 §19d0 的偏差理由**：§19d0 的「四卡、world_size=4、全局 batch 4096」针对**新机制候选**；
本 issue 是历史配方的**诊断复跑**，验收明确要求「原 batch/DDP 均不改变」。历史运行是单进程
（world_size=1, 每卡 4096），改成四卡会把 per-rank batch 变成 1024 并引入 DDP 梯度平均，
轨迹不再可比。故按历史单进程复跑，并在 VERDICT 记录该判定。

## 历史证据

|项|值|
|---|---|
|历史 Stage2 总耗时|1384.95 s（约 23 分钟，3000 epoch）|
|历史最终碰撞率|0.0370928（raw_unique 23,675）|
|历史 Stage3 `test_recall@10`（Nov-09 运行）|0.0633019（n_eval 57,439）|
|issue 引用的历史 Recall@10|0.0637372|
|差异|issue 引用值与本机现存 `test_final.json` 差 4.4e-4；按 issue 要求「历史 Recall 仅引用既有证据」，两者都记录、不冒称新测，并作为 C1 的一项待澄清事实|
