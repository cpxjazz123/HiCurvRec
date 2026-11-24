## 最终对账：实验完成，负结果，不采纳

代码与主体证据已提交并推送：`25a5164c246fa096e9703f3e3616c9c83a28ad94`（当次本地/远端 main SHA 一致）。最终裁决：`results/research/issue252/VERDICT.md`；14 条原文逐项核对：`CHECK_3.md`；科学结果与协议限定：`REPORT.md`；性能报告：`PROFILING.md`。

- 正式 Stage2 `issue252_formal_r2`：3000 epochs，四卡，global batch4096，seed42；合法 SID/checkpoint 逐项重新编码一致。
- 正式 Stage3：150 epochs，冻结 T5 协议，57439 条最终 test；独立逐用户复算 R@10=`0.0492870697609638`，未过严格0.065采纳线。不重训追分，不将无同源凭据的历史 TIGER Recall 混入比较。
- 完整全量硬q/连续z几何诊断均完成；每种表示28391948有效三元组。精确四项归档 `r2:genrec/100/`，100文件逐一下载核对SHA。
- 最终当前源码公共 MCP entry PASS + behavior_graph PASS；凭据 SHA256 `f7d24eb30580fd1246b071c1520023ef64d94aa7e7d86092db8e3d2a5976e015`。193 deterministic + 11 reference + 2 mutation suites，15个指定变异全捕获，0失败/错误/跳过。
- 当前实际四卡分模块 loss/VJP/梯度/参数更新归档 `MODULE_CHECKS.json`，SHA256 `97fbb5e0aff5e2315acb60fb9bbd4dac1627156e6ab8341b4124575454e9605f`；先前版本按原SHA保留，不以旧证明充当前源码。
- 独立完整复算命令退出0：`RECONCILIATION.json` SHA256 `d6cf1cd6520eb06e36bfe8803468c41835bb89de25315a38635e31d3bdf752e8`；源码/数据/结果/四个测试分片/归档身份均核对。

结论只对该完成的单臂成立：**未达到收益阈值，不能支持“混合曲率更适合、不同graphlet需要不同几何偏置”**，也不能据单臂负结果普遍否定混合曲率。

不隐藏资格限制：原0.5s要求未达；只按事前记录的owner0.8s及最后一级Sinkhorn修订交付。公共验证为事后调用、分层记录晚于Stage3启动但早于读分、正式Stage2 JSONL缺rank PID、旧Conditional G3定义缺失，均在REPORT/VERDICT明确保留，不回填、不冒称原始未修订全部协议零偏差。

关闭前的最后动作仅是发布这些最终裁决元数据、核对推送后的main SHA并关闭Issue；没有剩余实验、实现或正式训练待做，也不因负分追加实验。
