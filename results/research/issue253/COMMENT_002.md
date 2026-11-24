Infra前置完成：24个旧机制已全部移动到 `stage2_RQ-VAE/_archive/`（18个git mv、6个未跟踪目录mv），保留baseline及_archive；搬动前无训练/验证进程。证据 `archive.log`。新候选 `TIGER_three_expert_attention_rq` 只从冻结TIGER复制，15个源码/配置文件初始逐字节一致；#252实际执行源码仅作为只读机制参考，差异仅执行路径适配器。

资产核验91/91 PASS（manifest88项+train/valid/test），包括完整v9 CSR、节点特征与28,391,948目标；没有重复造资产。逐项path:sha256见 `results/research/issue253/asset_audit.json`，manifest SHA `8c58d34521181925df2e66dbe01f64b7150fb56a1eab028f59e8b4832c381ae6`。四张L40S空闲、50201/50202无监听。

已在生产修改前冻结 `CONTRACT.md`：三专家native指标外，#253和#252均以#252冻结尺度、相同完整目标/参考集统一复测三/五专家固定度量；同一步共享只复用动态token/gate，不合并量化总体。历史可变工具路径迁移仍待完成，不将该项提前判PASS。
