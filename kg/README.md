# GeneRec KG — Knowledge Graph of Verdicts

Phase 1 pilot: 从 2 个 verdict 抽取的 entity + relationship, 构建成 NetworkX 图.

## 目录结构

```
kg/
├── schema.json              # 实体类型 + 关系类型定义 (12 entity types, 25 relationship types)
├── extracted/               # 手动抽取的 verdict JSON
│   ├── verdict_task334_issue43.json   # Issue #43 Gate 2a GO 端点
│   └── verdict_task350_issue62.json   # Issue #62 Arm C Stage 2 NO-GO 端点
├── build_kg.py              # NetworkX 构建 + matplotlib 可视化
├── kg_graph.json            # 图序列化 (NetworkX node-link format)
├── kg_statistics.json       # 节点/边统计
├── kg_visualization.png     # 图可视化 (matplotlib)
└── README.md                # 本文件
```

## 当前 KG 状态 (Phase 1)

| 维度 | 值 |
|------|-----|
| Verdict 数 | 2 (task334 + task350) |
| 实体节点 | 46 |
| 关系边 | 50 |
| 实体类型 | 11 (Verdict, Issue, Task, Method, Hyperparameter, Metric, RootCause, Decision, Stage, Gate, Hypothesis) |
| 关系类型 | 15 (top: Task--implements-->Method, Verdict--evaluates-->Issue, Verdict--passes-->Gate 等) |

## Schema (kg/schema.json)

### 实体类型
- **Verdict**: 评估/决策文档 (verdicts/task<N>_*.md)
- **Issue**: GitHub issue 跟踪研究方向
- **Task**: 具体执行单元
- **Hypothesis**: 可测试的假设
- **Method**: 算法/技术
- **Hyperparameter**: 超参设置
- **Metric**: 评估指标
- **RootCause**: NO-GO 根因机制
- **Decision**: GO/NO-GO 决策点
- **Stage**: 流水线阶段 (Stage 1-4)
- **Gate**: 多阶段实验检查点 (Gate 0-4)
- **Paper**: 外部 paper 引用

### 关系类型 (top 15)
- Verdict--evaluates-->Issue
- Verdict--evaluates-->Task
- Verdict--tests-->Hypothesis
- Verdict--supersedes-->Verdict
- Verdict--identifies-->RootCause
- Verdict--achieves-->Metric
- Verdict--makes-->Decision
- Verdict--passes-->Gate
- Task--implements-->Method
- Task--configures-->Hyperparameter
- Method--variant_of-->Method
- Method--cites-->Paper
- Issue--depends_on-->Issue
- Hypothesis--refuted_by-->Verdict
- Stage--precedes-->Stage

## Phase 1 vs Phase 2 路径

### Phase 1 (当前) — 手动抽取
- ✅ 实施就位 (2 verdict, schema, build_kg.py)
- ✅ KG 可视化可用
- ❌ 扩展性差 (200+ verdict 都要手填)

### Phase 2 — DeepKE 自动抽取
- ✅ DeepKE 2.2.7 已装在 `/home/wlia0047/.conda/envs/deepke` (Python 3.9 + torch 1.11)
- ❌ cnSchema 不匹配我们域 (cnSchema = LOC/PER/ORG, 我们 = Issue/Verdict/Method)
- ❌ 需要 fine-tune 或 prompt 适配
- ❌ 英文 verdict 支持弱 (DeepKE 中文优化)
- **结论**: DeepKE 适合做中文 KG, 我们 verdict 混合中英文, LLM-based 抽取可能更合适

### Phase 2 备选 — LLM 抽取 via llm_client.py
- 走 MiniMaxAnthropicClient 零样本抽取 (per AGENTS.md Rule 8/9)
- 完全可控 schema (在 prompt 里定义)
- 双语支持 (中文/英文 verdict 都行)
- **建议**: Phase 2 优先 LLM 路径, DeepKE 作为 backup

## R7 / R11 / R15 合规

- ✅ R11.5 自主决策 (DeepKE 装好但用 cnSchema 不匹配, 暂用手动 KG)
- ✅ R15 (后续 commit + push)
- ✅ 不动 upstream verdict 文件 (只读取)

## 后续路线图

1. **Phase 2A**: LLM 抽取 50+ verdict, 扩展 KG 到 200+ 节点
2. **Phase 2B**: KG 嵌入 + 相似度搜索 (找出类似 NO-GO 模式)
3. **Phase 2C**: KG-driven research recommendation (新 issue 提案自动 link 类似 verdict)
4. **Phase 3**: KG-LLM RAG (query KG 来辅助 verdict 写作)

---

result: KG Phase 1 pilot — schema (12 entity + 25 relationship types) + 2 manually extracted verdicts (task334 Issue #43 GO + task350 Issue #62 NO-GO) + NetworkX graph (46 nodes, 50 edges) + matplotlib visualization. DeepKE installed but not used (cnSchema mismatch with our Issue/Verdict/Method domain). Phase 2 建议走 LLM-based extraction via llm_client.py.