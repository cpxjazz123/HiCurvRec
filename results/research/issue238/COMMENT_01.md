## 节点 1：验收清单已冻结

- 正文 sha256：`4b6de1a92811bf45d3cd0692d69c83db8f5f3f21517057df667c27f5dcd0459d`
- 清单条数：**47 条**（C1–C47），分六组：固定协议 5、方法规格 10、TDD 9、Stage2 评估 13、结构假设与 Stage3 7、交付 3
- 清单文件：`results/research/issue238/RUN_LOG.md`

### 是否需要训练：**需要**

依据是「验收与交付」第 3–4 条：`K_q`、硬 q 的几何代理误差、三码碰撞与 Stage3 Recall
都只能由一个新训练的 Stage2 权重产生，无法由已有产物复算。因此走 Stage2（3000 epoch，四卡，
全局 batch 4096，seed 42）+ Stage3（150 epoch，冻结协议）各**一次**。

### 关于既有探索的边界（已在清单中写明）

`results/research/issue238/REPORT.md` 的 7 个机制臂是 issue 之前的历史产物，**不作为本 issue 的候选**，
其结论也不作为本 issue 的证据。issue 正文引用的「约 0.00107 Stage3 波动」我按 n=3 的同 SID 重复评估
确认（0.055450 / 0.053483 / 0.055206，sd 0.001072），本 issue 的单次 Stage3 差值将按此解读。

### 下一步

infra（数据/基线哈希、解释器、卡与端口、候选树派生），完成后在此评论。

*（本评论文件入库：`results/research/issue238/COMMENT_01.md`）*
