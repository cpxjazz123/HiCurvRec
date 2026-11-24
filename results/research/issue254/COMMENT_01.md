冻结 #254 的 9 条验收，issue 正文 SHA256 `ce03103cf5d97ff95a8b62cfe321cbbec5f4a5fd9bcbff8bee7a92c92e844dac`，原文保存在 `results/research/issue254/ACCEPTANCE.json`。无需训练：本任务只读诊断现有 #252/#253，禁止新臂/重训。

两臂原始四分片 top-10 商品/SID 和目标秩都在；搜索 beam=20，但持久化候选只有10，未命中只能记为 `>10`，不能虚构11–20的秩。恢复时发现初稿字段错配 (`KeyError: rank_252`) 和未完成身份校验；先写回归断言再修复。原 dump 未直接写 user/history，将通过同 SHA 的原始 split、经核验的分片规则、每行目标及规范 SID 重建身份后按复合样本身份连接，不声称 dump 包含未保存的字段。

证据/契约：`RUN_LOG.md`、`CONTRACT.md`、初始 `EVIDENCE.json`（后者将补齐正确的 Stage3 checkpoint 路径与来源校验）。无静态资产重建、无训练源码或权重修改。
