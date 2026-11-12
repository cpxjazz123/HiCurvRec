# ideas/_archive

跑过的机制方向归档在这个目录里：`ideas/` 下面只留还没执行的
（`result.status = "not_run"`）与本目录。

归档规则与实现只有一处：`agent-vis/archive-ideas.js`（`result.status` 不是 `not_run`
就搬过来，文件名不变；解析失败或缺 status 的文件原地不动并被报告）。

归档与待做同根，但互不串味：页面扫 `ideas/` 时跳过这个目录，所以归档之后它自然只
看到待做的方向；归档脚本同样跳过它，不会把归档过的东西再归档一次。

```bash
node agent-vis/archive-ideas.js --dry-run   # 看会搬什么
node agent-vis/archive-ideas.js             # 归档
```
