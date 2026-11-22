# round 2 预注册判读规则(在看结果之前写定)

冻结时间:round 2 治疗臂 epoch 615、对照 epoch 584(均未跑满)。
本文件在看任何 round 2 的 Δ 之前写定,用来防止"结果出来后调整口径"。

## 判据来源

只来自 issue §5(正文 sha256 `301c5a9a93d73ee778bbc7135522ddb8a7d5fe72adc6d640ca312748bfc7b0a0`),不新增、不放宽。

## 判读规则

| 条目 | 通过条件 | 否则 |
|---|---|---|
| C36 | 三结构 Δ_arm > Δ_baseline | FAIL |
| C37 | fork Δ_arm > 0 且 fork Δ_baseline < 0 | FAIL |
| C38 | 三结构 satisfy_rate_arm > satisfy_rate_baseline | FAIL |
| C40 | 三结构 (Δ_arm − Δ_baseline) 的 95% 聚类 bootstrap 下界同时 > 0 | FAIL |
| C41 | 三结构 Δ_arm > Δ_control | FAIL |
| C43 | 三结构 near_share_arm ≥ near_share_baseline | FAIL |
| C45 | collision_arm ≤ collision_baseline | FAIL |
| C47 | recon_arm ≤ 1.05 × recon_baseline | FAIL |

## 附加的、事先声明的读法

1. **噪声底**:用 `COMMENT_23` 测得的采样噪声(chain 0.0008 / fork 0.0089 / merge 0.0082)。
   任何小于该值的结构间差异,在报告里一律写"在噪声内",不写成改善或退化。
2. **主要判据是 C41(对打乱对照),不是 C36(对基线)**。理由已在 `COMMENT_16` 给出:
   round 1 的 fork/merge 相对基线改善与打乱对照不可分,说明跨父系列比较不归因。
   若 C36 通过但 C41 不通过,结论必须是"机制无效",而不是"机制有效"。
3. **若仍 FAIL**:失败层级归为"间歇监督 vs 连续监督"(COMMENT_22 已量化:同一对被约 8 步采到一次,
   而重构梯度每步作用)。不再次归咎于预算或覆盖率 —— 这两条本轮已修并有量化证据。
4. **不做的调整**:不调 `PREFIX_TARGET_SHARE`、不改 `MARGIN`、不增加 epoch、换 seed、不挑 checkpoint。
   §19g 明确禁止试值挑选;本轮只有一次正式运行。

## 若全部 PASS

进入 skill §6:核对代码已推送(§43)、本地远端同 SHA(§45)、`VERDICT.md` 无遗留 FAIL、每个 C 的 PASS 已评论,
然后关闭 issue。Stage3 不在本 issue 范围(§5 明确"本 Issue 不要求启动 Stage3")。

## 若 FAIL

按 skill §5:Issue 保持 OPEN,`VERDICT.md` 写明失败层级与下一步假设;按 §7 判断机制主张是否在数学上不可满足
(目前测量显示**可满足**,故属 §5 未达成,不属 §7)。
