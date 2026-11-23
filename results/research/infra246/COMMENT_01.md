## 节点 1:冻结验收清单与训练判定

- 验收清单 **C1–C37**,逐条带判据/阈值/数据/口径,写入 `results/research/infra246/RUN_LOG.md`(同时记录 issue 正文 sha256 `b457e25698d34d1188a24e29694c71217bc17135c486c796808fe26a0e5b5362`、起始 HEAD `dbe8e354`)。
- **本轮不训练**。依据 issue 原文「本轮只做 Infra，不训练 Teacher/GNN/RQ-VAE/Stage3，不重训 TIGER，不扫参、不多种子。允许对现有冻结 checkpoint 做只读推理、重导和指标复算」与 §8「不得擅自创建其他 Issue 或启动任何训练」。全部验收指标可由已冻结产物 + checkpoint 只读前向复算,按 skill §4 属「分析/契约型」。
- 工作范围:`tools/graph_infra/`(共享生产模块,不做 issue 私有复制) + `results/research/infra246/`(v2 资产与证据)。历史资产只读。

下一步:写 `CONTRACT.md`(公式/形状/dtype/边界/容差 + 论文与官方实现对照),然后**先让现行生产代码在反例上失败**(C5/C10/C12/C17/C24/C28 等)。
