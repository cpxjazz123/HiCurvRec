C8（含连续诊断补项）PASS：`/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 -m tools.issue251.run_evaluation` 实际退出0，67.10秒；硬q/连续z各独立访问13分片全部28,391,948合法三元组、24,326中心和完整组件参考，没有抽样/拟合评估尺度。固定初始tangent尺度0.18152852356433868；独立ambient几何与实际生产距离抽样绝对误差q≤1.5543122344752192e-15、z≤5.329070518200751e-15。

| 指标 | 硬SID解码q | 连续z |
|---|---:|---:|
| 曲率macro MAE | 2.6846592811951826 | 2.7511175786802333 |
| 曲率micro MAE | 1.744486568784263 | 1.7864154747500234 |
| 符号balanced accuracy | 0.3361141896046254 | 0.33632345539577274 |
| 固定距离绝对失真均值 | 1.25322821227839 | 1.1952485398035626 |
| decoder重建MSE | 0.0002802857134533353 | 0.00025372100661630407 |

双方均实测10个冻结strict held-out bank，各自使用真实坐标；标签不进入模型、不训练probe。bank无二元label，AUC诚实NA；报告距离分布、合法拓扑字段的Spearman及缺失数，不伪造负例。分符号误差和每商品target/predicted分布已保存。硬SID L1/L2/L3不同前缀256/3191/9959，三码碰撞0.5949485500467726，三个码本各用256码字；这是单微步诊断快照，超过Stage3的0.4门槛，绝不拿另一份入口SID的0.0468替代，也不启动Stage3或据此调参。

冻结实际TIGER SID SHA `025ad8bd5809ce6296047514a80a4fedd012f3145d184eb4babd51c76c140059`；其历史几何/重建NA，因为现有checkpoint不是同代SID，未冒用或重训TIGER。报告不宣称推荐/几何收益。

绑定：`EVALUATION_SUMMARY.json:0592fc47c94ab30d1dba3b4c9e2f7ffffdbf0fb3016778b10869db925c0ccaef`；完整每商品分布 `HARD_SID_EVALUATION.json:2d37d7001bb97fdbd2a3c0a84fdc1c7a26982e0426eb5d0686cb64d8fa871465`（生成数据不入库，摘要记录路径/大小/SHA及重算代码）；`EVALUATION.log:7d631d4de2cccd2ad81cf076b139c2b95e928188568e2108b77dd23ce7360d9f`。新评估器与runner SHA `7ed9a6c3b26fe121f0c0fea737b200fe4c738f812832e4b252945cf568f8fb83` / `2e5605196fda424f0c6b03f76ebdbf10fff24dad521fdb624c96e05b42e51803`；22项测试日志 `CONTINUOUS_PARITY_FINAL.log:6f30cff2ad985fc952cfb013d28b112a4e096a0fb711568f4e052a5d3928c12f`。模型checkpoint和SID仍为C2/C5/C6同一绑定，未因补报告而更新权重。
