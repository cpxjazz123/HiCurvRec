R01同一最终Stage3 checkpoint的外部validation已完整执行并退出0，57439行全部覆盖；四rank逐行保留身份/target_rank，独立从预测归约与Trainer原始sum一致。未训练、未选checkpoint、未重放test。

|指标|完整validation|
|---|---:|
|Recall@5|0.0381622242727067|
|Recall@10|0.057643761207541916|
|NDCG@5|0.025688765803069922|
|NDCG@10|0.03194281550571229|

checkpoint SHA `d8540c837dedffd8291615bb6d7578308f74f6469e3182887685a08571082a96`与本轮最终test相同；valid.parquet SHA `48a291d174654ffb014d4d023029600168c7bdffc25d2c1982c93ed422b353e9`不变。命令为Stage3解释器`-B -m tools.issue256.external_validation`；证据`rounds/r01/recommendation_validation/VALIDATION.json:1245fb69ca0c958d85d63642c15666370f6e1df075d51f1d4dc8683b7a2e19ff`。消费者是test前冻结的36de17c3代码，真实GPU路径完成，不以先前两个回归代替。

R01的R6已完成（3000/150、精确SID、独立最终指标、完整validation及R2/102）。R@10目标仍失败，进入R7/R1/R2；不关闭#256。下一具体改动只依据训练/validation侧诊断。
