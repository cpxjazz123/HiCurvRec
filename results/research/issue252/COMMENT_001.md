已冻结原文14条验收条件，正文SHA `3e9b8ca4018cd9d0a43648efc42cd6a12ef27bc892fdb981590030bea37849eb`，见 `results/research/issue252/ACCEPTANCE.json` 和先于实现的 `CONTRACT.md`。

本轮**必须正式训练**：issue目标及关闭条件明确要求一次四卡Stage2 3000 epochs、其同源合法SID的一次Stage3 150 epochs/final test与R2归档；#251的代码、微步或smoke不能代替。正式派发前还必须通过真实图/数学/梯度/DDP/入口准入，并在global4096、seed42下实测连续至少30个完整epoch，含结构更新摊销的平均墙钟<=0.5秒；不把旧训练计时边界当新门槛。

已先读assets/与v6 manifest（68项，SHA `23a0a6e2c28795c9b768d702e048c5bbd1fe31f749ad0ed3860709f2ef457463`），正确二进制资产不重复构建。owner在#252已明确授权合法图契约修复和预先声明的训练估计器，因此不会沿用#251的旧合同阻塞直接收尾。初步来源研究表明62501边差异来自#246结构用户留出；将从原始train独立可执行复算并逐类对账后才注册具名可见图。当前4张L40S均空闲、端口50202/50201空闲；尚未派发GPU作业。
