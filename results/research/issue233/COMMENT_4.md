## 补充:C16/C31 的记录缺口可以用已有产物补上

上一轮把 C16(梯度范数与夹角)、C31(裁剪前后梯度/latent 位移/硬码变化)标为 PARTIAL,理由是
`graphlet_stats.json` 在联合路径下未写出。**这个缺口可以离线补齐**,不需要重训:

1. `stage2_RQ-VAE/TIGER_export_matched_graphlet_rq/rqvae_best.pth` 是最终 checkpoint;
2. 用它在冻结的监督 draw 上重放一次结构项与碰撞项的前向/反向,即可得到 `dL_g/dθ`、
   `dL_c/dθ`、`dL_RQVAE/dθ` 三者的范数与两两夹角;
3. 用同一 checkpoint 与 TIGER checkpoint 各做一次 50 步受控更新,记录 latent 位移与硬码变化率。

这与训练时的逐步记录不完全等同(后者含数据顺序),但能把"梯度是否真的按预算分配"这件事
**从断言变成测量**。本轮补做,结果写进 `gradient_audit.json`,并如实标注它是离线重放而非训练日志。

同时补 C34:在同一个重放里用 `all_reduce` 模拟四卡,确认预算分母的全局版本与本地版本的差异量级。
