# Check round 2 — source-integrity repairs

Issue body SHA-256: `eb05f12aee33f42cfff92568e925e00d4810fce6d78db807a31082611528af2d`. Live body was reread; criteria unchanged.

## Verbatim issue acceptance

- [ ] C1 审计 CLAUDE.md、assets/manifest.json、#249 接口和冻结 TIGER 源码。冻结数学契约、候选树边界和验收清单。已有资产按哈希复用，不重复构建。
- [ ] C2 实现多子图 GNN token 接口：方向、尺度、节点/中心索引、mask、重叠邻域、孤立点、批量/分片一致、固定全局商品身份。报告每商品 token 数量与覆盖；任何新增静态资产保存至共享 assets/ 版本目录，更新 manifest，验证重载。
- [ ] C3 实现五专家 Cross-Attention 与共享 score MLP。测试两个归一化轴、mask、padding 无影响、子图输入顺序等变且输出不变、单 token 退化、重复商品一致、不同 Query 的可区分性及可训练性；用手算 Q/K/V 样例独立验证，禁止仅凭权重形状通过。记录逐专家子图注意力与最终几何权重，二者不得混淆；不假定初始 Query 已有几何语义。
- [ ] C4 实现五几何代价并接入原 Sinkhorn/硬量化；测试单专家、混合、曲率关/行为关廉价接线开关，不训练消融。受控修改实际 attention/门控输出必须能改变代价并在构造边界样例中改变硬码字；真实 q 与 SID 解码完全一致。
- [ ] C5 实现可微截面统计 + 统一 Loss，冻结目标与量化预测严格对齐；提供 28,391,948 个合法三元组及全参考集合的分块、流式、内存、DDP 和梯度方案，不能拿 batch 内参考点冒充全参考。不擅自抽样或替换目标；测量实际成本及潜在阻塞。
- [ ] C6 真实小 batch 验证原 Loss 与曲率 Loss 的单独梯度：Encoder、码本、GNN、Q/K/V、score MLP 和 Decoder，记录未加权/加权范数、梯度方向、裁剪前后及参数更新。验证受控微步后权重/代价/硬 SID 的变化；切断码字代价或冻结 attention 的负向回归必须失败。可微路径有限差分对照，硬索引不要求有限差分等于 STE。
- [ ] C7 TDD：先失败测试再实现。执行数学、真实入口、四卡 DDP、梯度、导出、数据泄漏、数值稳定、碰撞、温度饱和、分块等价、checkpoint→SID 重导出一致性与 #249 回归；遵守仓库准入流程。仅允许验证性短运行，不派发正式 3000 epoch。
- [ ] C8 固定独立评估：硬 SID 解码 q 与连续 z 分开评估。报告曲率宏/微 MAE、分层误差、符号 balanced accuracy、商品曲率分布、固定度量距离失真、held-out 结构保真、前缀共享、三码碰撞、码本占用、重建误差。冻结实际 TIGER SID；历史几何指标不可溯源则 NA，禁止重训或冒用 checkpoint。
- [ ] C9 保存数学契约、候选代码、验收日志、配置、必要证据及资产 manifest；统一保存并验证重载。逐项报告 PASS/FAIL/NA、真实模型执行证据、显存/耗时与 SHA；提交并核对远端。



## Executed checks

- Candidate45 tests + independent evaluation/readonly19 tests: PASS, command exit0; `UNIT_BOUND.log`.
- Independent geometry identities, hand answers, attention axes, masks, controlled hard-selection intervention and finite differences: PASS.
- All9 mutations detected by specified assertions, no crashes counted: PASS.
- Full #249 regression79 tests +24 module checks: PASS, `REGRESSION_249.log`.
- Review4 findings: repaired; red and green evidence retained.
- Actual four-GPU microstep/full-target backward, true-entry MCP smoke and independent real-output evaluation: NOT_RUN in this round; separate execution is in progress.

Overall: NOT_PASS until actual runtime evidence is complete. No criterion is relaxed; no formal training.
