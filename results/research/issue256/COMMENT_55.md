R02新门控回归在highest下已实际PASS，但完整suite暴露原有“两个相同实例”硬输出断言失败。原进程未保存计数器，因此不追认其原因。另做确定性最小复现：真实baseline副本安装框架后，调用实际VQLayer._indices，便宜选码[0,1]、同owner下一次覆盖[2,3]、另一个owner独立[0,1]均通过；销毁量化器后，指定 `destroyed quantizer must release its rotating-coverage history` 红（1failed，4.09s）。

旧floor以永不释放的整数id存历史，真实生命周期缺陷已被独立证明。现在框架/两类安装模板及当前候选统一改为弱引用layer对象作key；存活owner的覆盖策略不变，无整数兼容缓存，历史只读树未动。正式1024/256和全目录导出处于可行floor分支，本就不读取旋转历史。

`FLOOR_LIFETIME_RED_RECEIPT.json` sha256 `59da11aaaea5f67a0e13c9a865b44bf1fe182b97f4c444c1b9d9c1838388583d`；`FLOOR_LIFETIME_REPAIR.md`记录边界。已只重载omp的stage2/genrec子进程，并实际调用status和缺证据guard，见MCP_RELOAD.json。新回归及原安装幂等性加入完整source-bound suite，bg_513正重新检查；仍未正式训练。
