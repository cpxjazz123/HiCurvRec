# ideas_archive

跑过的机制方向归档在这里：`ideas/` 只放还没执行的（`result.status = "not_run"`）。

归档规则与实现只有一处：`agent-vis/archive-ideas.js`（`result.status` 不是 `not_run`
就搬过来，文件名不变；解析失败或缺 status 的文件原地不动并被报告）。页面只读
`ideas/`，所以归档之后它自然只看到待做的方向。

```bash
node agent-vis/archive-ideas.js --dry-run   # 看会搬什么
node agent-vis/archive-ideas.js             # 归档
```
