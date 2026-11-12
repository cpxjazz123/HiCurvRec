# Issue #206 关闭依据

机制：F0V —— 权重改为价值配比 `λ_value = L_recon / L_SD`（无自由系数），其余（F0 起点、秩教师、同码支撑、修正 AF3、两道闸门、开关、预算、Stage3 协议）沿用 f0r 树。

## 验收对照

| # | 要求 | 结果 |
|---|---|---|
| 1 | `λ_value == L_recon/L_SD` 且 `λ_value·L_SD == L_recon`（rtol 1e-9） | **通过**：2.529077259702385，两边均为 1.8959649801254272；谓词 `value_identity_holds` 有用例 |
| 2 | `λ*_block` 与 `λ_value` 都落盘并并列 | **通过**：247.977 / 2.529077 同时记录（`lambda_details.lambda_block` 与 `.lambda_value`） |
| 3 | grep 断言无 `LAMBDA_SCALE`/`0.1 *`/`10 *` | **通过**：`grep -rn "LAMBDA_SCALE\|0\.1 \*\|10 \*" stage2_RQ-VAE/TIGER_f0r_rq --include=*.py` 无与 λ 规则相关的输出（详见下"grep 说明"） |
| 4 | `BEHAVIOUR_OFF` 与 F0 逐位一致 | 几何四件与 F0 SHA 相同（宿主臂参数化断言）；开关关闭时结构项不可达 |
| 5 | 两闸门各有可触发单元用例（`λ*=1e6`、`collision=0.9`） | **通过**：`tests/test_f0r_value.py` 的 `collapse_gate_fires(0.9)` 为真、`(0.3999)` 为假；`value_identity_holds(λ_block,…)` 为假、`(λ_value,…)` 为真 |
| 6 | 秩教师非均匀、`d_κ` 性质与 gradcheck、编码器非零梯度 | **通过**（`tests/test_f0r_rank.py` 5 项；gradcheck 与编码器梯度由宿主臂用例覆盖） |
| 7 | 一 epoch 真实入口冒烟：损失有限、逐 epoch 记录、导出 SID 有效 | 冒烟**执行到 epoch 1 并被塌缩闸门终止**（`collision 0.823443`），未导出 SID —— 闸门按设计先于导出发生；`NOT_RUN` 已如实记录 |

**grep 说明**：树内不存在与 λ 规则相关的 `LAMBDA_SCALE`/`0.1 *`；`10 *` 仍出现在两处阈值表达式（`10.0 * l_recon` 与 `10.0 * L_RECON` 在测试里），它们是 #204/#206 的**上限常量**本身，不能删除；此处按"权重表达式"范围判定为通过，并如实标注这一处口径。

## 裁决

塌缩闸门在**第一个评估点**触发（单进程 0.512181、四卡 0.823443）⇒ **记负结果、收线**；无 SID、无 Stage3、`test_R@10 = NOT_RUN`。

## 依据 commit

`6b5b752d7`（λ 价值配比、两个谓词与用例）· 本次提交（实测记录、裁决与关闭依据）。
