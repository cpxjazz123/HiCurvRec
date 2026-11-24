完整validation配对诊断完成，退出0，57439逐行target/user身份及四分片预测与固定Parquet一致；已有R/NDCG重新核对不变。54128个top10 miss按最深命中语义前缀互斥分解：L0=41397（miss的76.4798%）、L1=11132、L2=1514、L3=85；第四碰撞token明确排除。验证target出现在其seen_history的行数为0，因此该验证总体不支持“gold已见而被过滤”解释。

这表明粗前缀失败占主要验证错误，但不证明门控/梯度是因果根因。保留逐用户/商品配对NPZ、训练侧degree/reconstruction与各级训练曲率协变量；训练曲率仍是样本内，不冒充held-out几何。没有读取recommendation test或结构test错误行。

源代码使用五个手算前缀答案（含同三码不同第四token）验证，相关消费者共3测试PASS；这不是新的机制训练准入。完整实际数据证据`rounds/r01/recommendation_validation/VALIDATION_DIAGNOSIS.json:2913227ac2617bc428b46ab9261ae1fe3c5f59923ac2813b03faef3aca50f4a0`，命令Stage2解释器`-B -m tools.issue256.validation_diagnosis`。当前所有训练/验证/诊断进程均已正常退出，接下来先按规则归档R01，再登记新机制假设。
