# Issue #240 — VERDICT

- issue 正文 sha256 `1c0798dc083e400ded1f2a4c01b9562b88b9a7cbaf84a1691553de1a10c42539`
- 清单 18 条（`RUN_LOG.md`），证据 `REPORT.md` / `report.json` / `raw/steps.jsonl`

| 编号 | 要求 | 实际 | 证据 | 判定 |
|---|---|---|---|---|
| C1 | 核对历史成功版本源码/数据/配置/路径/ckpt/SID/历史结果哈希 | 全部记录；issue 引用 recall 0.0637372 与本机 0.0633019 差 4.4e-4，如实并列 | `INFRA.md` | **PASS**（含未决差异） |
| C2 | 冻结超参不变，核实实际每卡/全局 batch | `ddp1`, world_size 1, per-rank 4096, item 4096, pair 2048 —— 与历史逐键相同；发现单卡直跑分支会放大 4 倍，已避开 | `INFRA.md`、`train_start` 记录 | **PASS** |
| C3 | 行为权重 ≤1500=0、线性到 0.2；核对三层行为项与梯度路由 | 权重曲线复算一致；**三层中只有 L1 有梯度，L2/L3 恒零**（STE 断开残差） | `REPORT.md` Q6/§3、`raw/steps.jsonl` | **PASS** |
| C4 | 只运行一次原配方诊断复跑；不重训 TIGER/不扫参/不多种子/不改损失/独立目录/不跑 Stage3 | 3000 epoch 单次；结果在 `..._diag/resample`；未跑 Stage3 | 日志 + `diag_manifest.json` | **PASS** |
| C5 | 诊断无副作用；小批量与四卡 smoke 比较开关前后 loss/grad/param/SID | SID 与参数 sha256 在 single / **ddp1** / ddp4 三模式下开关前后**逐位一致**；off 重跑同样一致 | `equivalence/equivalence.json` | **PASS** |
| C6 | 每步记录 epoch/step/rank/lr/权重/batch/各 loss | 18,000 行 = 3000×6，无缺漏，含 `rank`/`world_size` | `raw/steps.jsonl` | **PASS** |
| C7 | loss 拆分（重建/commitment/codebook 逐层与总计/原始完整/各层行为/加权行为/总） | 全字段；系数按实现：逐层量化项 **1/3**、commitment β/3、行为 w/3 | `report.json`、`REPORT.md §2` | **PASS** |
| C8 | 各模块梯度 L2/RMS/max/nonzero；None 与真零区分；两类比例；分母近零标注 | 全部字段；`n_none`/`n_zero_tensors` 分列；比值有 `denominator_below_1e-12` 守卫 | 同上 | **PASS** |
| C9 | 方向余弦/点积、冲突率与强度；零范数 NA | 8 组方向；行为 L2/L3 因零梯度记 NA；负余弦率 42.5%，强度 0.059 | `REPORT.md` Q3 | **PASS** |
| C10 | 裁剪前/后范数、系数、是否触发、各模块；加权和 = 实际 backward（DDP 同步与误差） | 0/18000 触发；identity 中位 7.9e-04；ddp4 时 rel_error 1.06 = rank 离散度（已解释，未冒充全局） | `REPORT.md` Q4/§1 | **PASS** |
| C11 | AdamW 前后各模块 Δ L2/RMS/max/相对率；码字/探针位移与分配变化率 | 全字段；探针每 epoch（3000 行），码本统计每 epoch | `report.json` Q5/Q6 | **PASS** |
| C12 | 有限性/爆炸/消失/退化/异常 batch；不屏蔽 | 训练器自带非有限检查；无爆炸（max 0.1326）；记录早期塌缩（epoch 50 仅 67/256） | `REPORT.md` Q6 | **PASS** |
| C13 | 关键节点单列；每 epoch 汇总；禁用期不混反事实 | `key_steps.md` 覆盖首步与 issue 的 10 个 epoch；`epoch_summary.csv` 每 epoch 一行；禁用期行为字段为 None | 两文件 | **PASS** |
| C14 | 逐 step JSONL、epoch CSV、曲线、日志、配置与哈希、REPORT；标存储与开销 | 全部存在；开销 **759.8 s / 1449 s = 52.4%**；raw 约 100 MB | `REPORT.md §6` | **PASS** |
| C15 | 重建 MSE、码本利用率/熵、间距、SID 碰撞、行为关系；与历史同口径并说明是否重现 | 重建 1.39e-4；L1 237/256、熵 5.36、间距 0.1046；碰撞 0.03945 vs 历史 0.03709；**SID 未逐位复现**，已说明原因 | `REPORT.md §5` | **PASS**（记录未复现） |
| C16 | 回答六问，区分实测/假设，给调度参考不预设最优 | 六问逐条；实测与解释分开；给出 ~7e-4 / ~1e-2 的等量级权重参照 | `REPORT.md §3/§4` | **PASS** |
| C17 | 不改现有训练机制、不建额外 Issue、不训练其他模型 | 诊断树与基线 diff 仅 46 行新增 + 1 行捕获返回值；基线只读未动；未建 issue | `git diff` | **PASS** |
| C18 | 若无法保证无副作用或原配方无法复现，报告阻塞 | 无副作用已证明；配方复现到超参/批次逐键一致，但 SID 未逐位一致（环境差异，已记录） | 同上 | **PASS** |

**整体判定：18/18 PASS。** 未测 Stage3 —— issue 明确「本轮不运行 Stage3」。
