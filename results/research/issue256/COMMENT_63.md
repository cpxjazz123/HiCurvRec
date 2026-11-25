R02生成入口ownership回归已绿（真实freeze/load，1passed、0.20s）；完整145/64/18逐例units、全部四卡Loss/更新/native/epoch51/inclusive诊断和machine entry/graph均实际通过。bg_515仅在最后凭据收集失败：steady_profile found2。

根因已证实是collector只比较当前candidate的子集，遗漏“旧源多出已删除文件”及控制器缺失/变化；并非要按时间挑一个profile。新回归3红/1绿：删除源残留、缺控制器、改控制器都暴露旧选择错误，两个真正完全同源的记录仍必须拒绝。

`PROFILE_IDENTITY_FAILURE.json` sha256 `cfcf91b4dd371e3f434c3df80c56d82daf94829e8cfdc9eb6c6684d21d7cb94a`；`FORMAL_ENTRY_GREEN_RECEIPT.json`绑定ownership实测绿色。现在所有角色共用包含profiling依赖的完整inventory，profile前/后必须与它全等；去掉分叉指纹和子集匹配，不按mtime/速度选。bg_516重新完整验证所有已变更的source-bound证据；仍无正式worker或R02 test结果。
