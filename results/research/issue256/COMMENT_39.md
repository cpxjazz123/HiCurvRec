R01 结论对账及 CHECK_4：15 PASS / 2 FAIL / 0 NOT_RUN。C1/C17严格目标未过：完整150epoch、57439 test的R@10=0.05273420498267727；阈值>0.065，差0.01226579501732273。本轮不采纳，Issue256继续OPEN，不追分重跑同机制。

完整验证R@10=0.057643761207541916；最终四卡全局诊断13/17组突破初始预算，最大9.0859271222；仅sharedScore原始梯度点积为负。局部Key/Query0方向结论不成立于全局平均；根因与Recall因果影响仍未知。

证据：`results/research/issue256/rounds/r01/CHECK_4.json` sha256 `9bf47d3563f35cc9063fde6c3103b9d343d4e08586d6bd92fa1b1caab317a36e`；`REPORT.md`、`VERDICT.md`、`ROUND_RESULT.json` sha256 `337f64a6ab11a5e85e8d84792452ae042179e0c60d3c87ca4cb16cbadf19c420`。完整源码/SID/checkpoint/召回日志已117对象SHA核验归档R2/102。已结束的活跃树完整归档，历史执行快照不变。

代码及本轮结果提交 `6cd32f054f139dccb8eee03f49afb08e59c50fa9` 已推送，main本地/远端SHA一致。下一轮具体修改仅由训练/验证诊断驱动；固定test反复访问的选择偏差已披露。
