## 节点 1：冻结 24 条清单；本轮**不训练**

- 正文快照 `results/research/issue245/ISSUE_BODY.md`，sha256 `bd0c9caeddf12feb661b055cbbf0043aacf34e072da415a5fe8caa83a4d97f5a`
- 清单 24 条见 `results/research/issue245/RUN_LOG.md`

**训练判定：不训练。** issue 原文「**不训练 Teacher/GNN/RQ-VAE/Stage3、不重训 TIGER、不做参数扫描或多种子**」，
验收「**不要求 Recall 提升**」「不得擅自创建其他 Issue 或启动训练」。因此按 skill §4 跳过训练节，
全部产出是可复算的公共模块、冻结资产与独立验证。

### 交付计划（按 issue 的 7 节）

1. 公共模块迁到 `tools/graph_infra/`（新路径，理由：§94 已把 `tools/research/` 收窄为只保留 Stage2/Stage3 启动；
   infra 不属于它，也不属于任何单个实验目录，避免各 Issue 私自复制）；
2. 冻结资产写 `results/research/infra245/assets/` + `manifest.json`（名称/路径/schema/形状/哈希/生成命令/合法用途）；
3. 复用 #243/#244 已验证的实现与测试（几何算子、诱导子图、pair bank、指标），迁移后**重新跑全套并重新绑定哈希**。

*（`results/research/issue245/COMMENT_01.md`）*
