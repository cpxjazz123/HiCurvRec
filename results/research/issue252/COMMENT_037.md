## 正式 Stage3 完成：负结果；精确 R2 归档完成

本次 `issue252_formal_r2` 的合法 SID 已完成冻结四卡 Stage3，150/150 epochs、最终 test 完成、57439 条评估记录。真实结果：

| 指标 | 数值 |
|---|---:|
| R@5 | 0.03278260415397204 |
| R@10 | 0.0492870697609638 |
| NDCG@5 | 0.021616071827379253 |
| NDCG@10 | 0.02692583040624563 |

R@10 未达到仓库科学阈值 0.065。这个实验不能支持“混合曲率更适合、不同 graphlet 必须使用不同几何偏置”；单臂也不能替代同容量单几何对照。不为追指标重跑正式训练，不把阈值当作有同源凭证的 TIGER Recall。

完整归档命令已成功：`python3 results/research/issue139/upload_stage3.py TIGER_gnn_attention_sectional_rq --tree results/stage2_RQ-VAE/_runs/issue252_formal_r2/TIGER_gnn_attention_sectional_rq/src`。

返回：`100 files, recall keys 23 -> r2:genrec/100/`；上传后已按脚本逐文件下载核对 SHA。四项顶层布局保留：机制源码目录（含完整绑定源和 tar）、SID、Stage2 checkpoint、含全部最终指标与两份完整日志内容的 recall.json。日志 `R2_UPLOAD.log`，索引 `results/research/issue139/r2_ledger.json`。SID 为 `037766704565c5143ab721fd5ca5f61bbba3e800baa9dd561d3ae5601987cff2`。

运行修复与完整曲率评估已提交/推送 `0578be56f618597ef9cc05620eef9daa765c91e5`，当次本地/远端 main SHA 一致。commit_guard 的全工作区默认预算被已有历史证据占用（147524 行历史 SID、14501 行旧 graphlet 统计等）；仅为这些保留证据显式使用 `--max-paths 1000 --max-lines 300000`，未改写/选入无关历史文件。当前生成的大表和重复诊断代由既有/补充 ignore 规则留在共享存储。

当前源的 C3 公共验证和最终独立逐用户/逐中心复算仍在收口：已冻结的新回归区分 beam=20 与实际报告 cutoffs=5/10；正在重新产生同源的诊断凭据（不是第二次正式3000-epoch训练）。原始事前凭据与时间顺序偏差保留；Issue 暂未关闭。
