## 节点 1：冻结验收清单（31 条），本轮需要训练两个模型

- 正文快照 `results/research/issue242/ISSUE_BODY.md`，sha256 **`0e47f69787342896ca8754668c2c3070abc79231a185349ec2e0c19bd587b480`**
- 清单 31 条（C1–C31）见 `results/research/issue242/RUN_LOG.md`

**训练判定：需要。**
- 步骤 4：「只进行一次 Teacher 正式训练」→ 一个 Teacher；
- 步骤 8：「只训练一次 Stage2 3000 epoch」→ 一个 RQ-VAE；
- 步骤 1：唯一正式基线为冻结 TIGER，不重训；
- 步骤 9：达准入后 Stage3 一次。

**与 #241 的边界（本 issue 已明确）**：不依赖 #241 的 Teacher、不依赖 #237 的导出向量；
**不设下一商品 ID 分类、链接预测 Recall@10 门槛或正负样本分类 Loss**。Teacher 唯一监督是
图距离失真 `L_D = mean |(D_geo/g)² − 1|`，`g` 为可见图无向投影的无权最短路。
因此 #241 的失效模式（用结构特征预测具体后继）在本轮**不适用**，判定改为距离保真与独立上下文关联。

*（`results/research/issue242/COMMENT_01.md`）*
