## round 2 代码完成:两项缺陷都修了,15+16 个测试绿,1-epoch smoke 通过

### 修了什么

| 缺陷 | 修法 | 验证 |
|---|---|---|
| 预算按 `cost.mean()` 标定(0.0022×间距) | 按**码字间距**标定:同分位数匹配 `gap_q / grad_q`,再**clamp 到 ±1 个间距** | 实测 shift 最大值 5.7e-4 < 间距中位 1.36e-3;翻码 92/4096 |
| 只监督三方同批的三元组(7.05% 覆盖) | 改成**成对约束**(两个商品同批即可),保留 near/far 标签 | 约束数 104 → 931(9×);携带梯度的商品 289 → 1,010 |

### 两个新发现的边界条件(都已写进代码与测试)

1. **分位数匹配不能取到 0**:成对约束下梯度在 75% 的行上恰为 `0`,直接取 25% 分位的 `per_item` 会得到 0 并除零。改为只在**有梯度的行**上取分位。
2. **不 clamp 会失控**:不限幅时 shift 达到 6.4e3,而间距只有 1.4e-3,翻了三分之一的码 —— 那已经不是"表达偏好"而是"重写量化器"。clamp 到 ±1 个间距后翻码 92/4096。

### anti-collapse 闸门的一次真实拦截

`tools/research/stage2/anticollapse/patch_tree.py` 的 `unguarded_sites` 把 `gradient.abs().max(dim=1)` 判成"第二处码字分配"并拒绝启动。那是误报(它算的是梯度幅值,不是 `argmin`),但闸门是 fail-closed 的,正确处置是**改写我的代码**而不是加豁免:改成 `.amax(dim=1)`。这符合"闸门只能被满足,不能被绕过"。

### 测试

| 树 | 结果 |
|---|---|
| 治疗臂 `TIGER_graphlet_wide_l1_rq` | **15 passed** |
| 打乱对照 `TIGER_graphlet_prefix_shuffled_rq` | **16 passed** |

两臂的 `train_rqvae.py` 除 `ARM_NAME` 外逐字相同;`hard_prefix.py` 只多 `ARM_CONTROL` + `shuffled()`。

### 1-epoch smoke

命令 `ISSUE230_SMOKE_EPOCHS=1 python3.9 train_rqvae.py`,exit 0,新 SID 与 checkpoint 已写出(`smoke/TIGER_graphlet_wide_l1_rq/`,13:50)。

### 下一步

按 §19g 起 round 2 的正式 3000-epoch 双臂,再按 §5 对账。issue 保持 OPEN。

证据:`stage2_RQ-VAE/TIGER_graphlet_wide_l1_rq/tests/test_hard_prefix.py`、`.../losses/hard_prefix.py`、`results/research/issue230/smoke_round2.log`、`budget_diag.txt`。
