## C10 / C13：正式 checkpoint 的全量独立几何评估完成

命令：Stage2 解释器执行 `python -u -m tools.issue252.final_evaluation`，成功退出；日志 `FINAL_EVALUATION_BOUND.log`。本次只读 CPU 评估，不训练、不重新导出 SID、不读取 Stage3 分数。

每种表示均覆盖 **28,391,948** 条冻结有效三元组、24,326 个中心；完整原始连通分量作参考，未抽样、未拟合评估尺度。每种表示独立核对 334,050,193 个距离值，ambient 参考与生产度量最大绝对误差分别为 6.21e-11 / 2.48e-10（容差 1e-9）。

| 指标 | 硬 SID 解码 q | 连续 encoder z |
|---|---:|---:|
| 全量曲率宏 MAE | 1.9442331968 | 1.9511554393 |
| 全量曲率微 MAE | 2.6121500435 | 2.6194174267 |
| 分符号 balanced accuracy | 0.3428875919 | 0.3423078027 |
| 重建 MSE | 0.00008840025082 | 0.00008917173966 |
| degree-matched held-out hop-distance Spearman | 0.1921112541 | 0.1939767784 |

硬 q 的负/零/正符号召回为 0.07451254 / 0 / 0.95415024：不能把较低量化碰撞率解释为已学好正负曲率，更不能由此证明“混合曲率优于单几何”。末轮五专家平均门控中欧氏专家约 70.27%，只是该臂的诊断，不是单几何对照结论。

只比较冻结历史 TIGER SID：SHA256 `025ad8bd5809ce6296047514a80a4fedd012f3145d184eb4babd51c76c140059`，其三码碰撞率 0.1692764469。历史 TIGER 的同源 codebook/geometry/reconstruction/Recall 证据不足部分均 NA；未重训 TIGER，也未混用它的另一个 checkpoint。

完整报告 `FINAL_EVALUATION.json` 为 41,864,252 bytes，SHA256 `e832369d59ea9d8d31a917b274de81afbe38262c1027f274e0b469e8bfbfa3ba`，按既有规则留在共享存储，不入 Git；摘要 `FINAL_EVALUATION_SUMMARY.json` 与重算代码保留。

同时定位 Stage3 status 的真实监控缺陷：JSONL 超过 64 KiB 后，身份查找只读尾部而丢失首行 train_start，导致正在训练的 run 暂时无 progress。已补 `test_stage3_metric_identity.py`：两个用例先失败，再改为从文件头读身份、从文件尾读最新训练记录；`METRIC_IDENTITY_RED.log` / `METRIC_IDENTITY_GREEN.log`。不改正在运行的 trainer 或训练状态。
