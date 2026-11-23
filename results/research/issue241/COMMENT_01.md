## 节点 1：冻结验收清单（23 条），本轮需要训练两个模型

- 正文快照 `results/research/issue241/ISSUE_BODY.md`，sha256 **`ba50e87a914e212d894b6ca0a6f728b222d6af3a81e9569dec77d94920035c55`**
- 清单 23 条（C1–C23）见 `results/research/issue241/RUN_LOG.md`

**训练判定**：需要。依据——
1. 「Teacher 必须有明确训练目标，禁止使用未训练/随机 GNN embedding 的余弦作为关系标签」→ Teacher 必须训练；
2. 第三阶段「只训练一个候选」→ 新 RQ-VAE 必须训练一次；
3. C18 明确禁止重训基线，C22 仅在准入通过后跑一次 Stage3。

**两处会先做、做不过就停**：
- C6：Teacher 若在独立样本学不到超越度数/简单统计的信号，先报告并停止正式 Stage2；
- C16：结构项开/关等价性与「只更新 Encoder」不成立，就不进正式训练。

*（`results/research/issue241/COMMENT_01.md`）*
