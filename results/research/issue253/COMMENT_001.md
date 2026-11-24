冻结验收：17条原文checkbox，正文SHA-256 `0a35f9d2c84736fbb8db523acdea5cd822b70d3fc3d78c90d56f0f76020cff3a`。

必须训练，依据目标及§4：一次四卡Stage2 3000 epochs + 一次冻结Stage3 150 epochs和实际最终测试；不重训TIGER、不做多种子或调参筛选。性能目标约0.4/期望≤0.5s，未达只能在完整机制实测≤0.8s时继续，不能删2-hop或采样。

证据：`results/research/issue253/issue_snapshot.json`、`acceptance.json`、`RUN_LOG.md`。已先读取assets；下一前置动作是归档全部旧机制，保留baseline与_archive，完成前不开展新机制设计/复制。
