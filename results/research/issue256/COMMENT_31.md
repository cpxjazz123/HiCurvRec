R01正式Stage3已finished：150epochs、四卡/global16384、beam20、完整57439样本最终test，原训练进程已退场。实际test_final：

|指标|实际值|
|---|---:|
|Recall@5|0.034105746966346905|
|Recall@10|0.05273420498267727|
|NDCG@5|0.022554059459244956|
|NDCG@10|0.02852344807984429|

**没有达到严格R@10>0.065**，差0.01226579501732273；不会以R01负结果关闭#256，也不会同机制换种子/追分重跑。Stage2的非塌缩、重建及选择性前缀预测成立不等于推荐关闭条件成立；与历史#252/#253的分数差不是单因素因果证据。

固定最终文件：`results/stage3_T5Train/TIGER_module_budget_prefix_rq/logs/Amazon_2023_Instruments/Nov-25-2026_06-36-15/test_final.json`；`rounds/r01/FINAL_STAGE3.json`和`STAGE3_FINISHED.json`绑定精确source/SID/Stage2及Stage3 checkpoint与MCP终态。接着完成已冻结的独立原始预测归约、同checkpoint外部validation及四件完整R2归档，再用训练/验证侧证据进入R1/R2下一轮；不读取test错误商品来选改动。
