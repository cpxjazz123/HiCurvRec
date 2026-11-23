## 节点 1：冻结验收清单（18 条），本轮需要训练

- 正文快照 `results/research/issue240/ISSUE_BODY.md`，sha256 **`1c0798dc083e400ded1f2a4c01b9562b88b9a7cbaf84a1691553de1a10c42539`**
- 清单 18 条（C1–C18）见 `results/research/issue240/RUN_LOG.md`
- **需要训练**：依据「固定协议」第 3 条「只运行一次带观测的原配方 Stage2」。验收指标是逐 step 的
  梯度/裁剪/参数更新轨迹，只能由新权重产生；issue 同时明确「本轮不运行 Stage3」，故只跑 Stage2 一次。

### 已核实的三件事（issue 点名要核）

**1. 实际 batch 与注释不一致。** 源码第 410 行注释写「总 batch = 1024 * 4 = 4096」，
但常量 `BATCH_SIZE_PER_RANK = 4096`，历史 `train_start` 记录 `world_size=1`、
`global_item_batch=4096`。即**每卡 4096、全局 4096**，注释是旧的。按历史实际值复跑，不改动。

**2. 历史成功 resample 是单进程运行**：`mode=ddp1`、`cuda_visible_devices="1"`、`world_size=1`。
因此复跑也必须单进程 —— 四卡会把 per-rank batch 变成 1024 并引入 DDP 梯度平均，轨迹不再可比。
这与 §19d0 的「四卡」不冲突：§19d0 针对新机制候选，本 issue 明确要求「原 batch/DDP 均不改变」。

**3. 行为权重实现**：`behavior_weight(epoch) = 0.20 * clamp((epoch-1500)/1500, 0, 1)`，
与 issue 的「≤1500 为 0，线性升到 3000 的 0.2」一致。三层行为项 = `behavior_contrastive_loss`
对 `ACTIVE_LEVELS=(0,1,2)` 的逐层 contrastive CE 取**均值**（`train_rqvae.py:849-868`）。

### 一处需要澄清的历史事实

issue 引用历史 `test Recall@10 = 0.0637372`，而本机现存 `test_final.json`
（`Nov-09-2026_15-08-53`）是 **0.0633019**（n_eval 57,439），差 4.4e-4。
按 issue「历史 Recall 仅引用既有证据」，两者都记录、不冒称新测，作为 C1 的待澄清项。

*（`results/research/issue240/COMMENT_01.md`）*
