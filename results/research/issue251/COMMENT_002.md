检查轮1：候选、独立评估器及资产只读回归共60项测试 PASS（退出码0）；#249 回归79项测试、24项模块检查全部 PASS。9个变异由指定断言捕获，覆盖数值、结构和梯度路由。

- `results/research/issue251/UNIT_FINAL.log:8c4d4023186eb2aad28d96bea8483ac1a796068409dc6993f2068cc4743918fb`
- `results/research/issue251/REGRESSION_249.log:512482158ca319c21ee3be1f8d48b0c7317789becd335484275b46cfa624f152`
- NaN padding 在真实 attention 路径中复现后修复；`RED_PADDING.log:b00b14cce24d4772c0b91cefea80de545fc5e68c614565f1d3e23a4448081a2a`。
- 资产 `--verify` 不再重写/降级新版本 manifest；`RED_READONLY.log:6da86e708129c43172fa35daf69d9154bc1d7b584e6dbce4bddaa910daf7f964`。

当前不把 C5/C6/C7 判为完成：真实四卡微步、全量28,391,948目标反传与入口验证尚待执行。本轮无正式训练。
