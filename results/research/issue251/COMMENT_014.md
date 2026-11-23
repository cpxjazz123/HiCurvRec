C7 新发现的可修复工程问题已红→绿：MCP 原生请求300秒超时，但入口子进程仍继续；machine_admission 每次直跑 run_all，旧缓存写入还依赖已删除的 core.evidence，导致结果无法在重连后取回。未重启或中断仍在运行的入口 PID2456764，未盲目重复派发。

已改为持久化、源码/配置及共享 Stage2/GNN 运行时代码绑定的入口缓存；每次命中仍重新检查当前行为图，不能用缓存绕过图完整性；NOT_RUN 不缓存。系统 Python 的7项隔离回归全部PASS，覆盖重连、失败结果、配置变化、图检查器变化、共享GNN代码变化、可变图判决及NOT_RUN。红证据 `RED_MACHINE_CACHE.log`、`RED_MACHINE_IDENTITY.log`；绿证据 `MACHINE_CACHE_FINAL.log`，命令 `python3 -m unittest tools.research.stage2.tests.test_machine_cache -v`，退出0。

生产补丁 `tools/genrec-mcp/server.py`、独立回归 `tools/research/stage2/tests/test_machine_cache.py`。等待原入口完成后仅重载目标MCP服务，并经公开工具实测取回，不把单测当作真实入口通过。冻结图/规范图合同冲突仍不放宽。
