# #200 派发前验证

| 项 | 结果 |
|---|---|
| 训练器导入 | 通过；`SUPPORT_OFF=False`、`GEOMETRY_OFF=False`、`BEHAVIOUR_OFF=False`；`ANCHORS_PER_STEP = 63` |
| 验收用例 | **15 passed**（含"几何文件与 F0 逐字节相同"的参数化断言、开关关闭时机制不可达、AF3 三组图论、`codes_of == get_indices`、`A_κ` 性质与 detach、空支撑不产生 NaN、结构项到达编码器、κ 表可载入 >20,000 条） |

"起点唯一差异"断言的实际内容（`test_only_the_identity_and_the_structural_modules_differ`）：

- 新增文件被限定为 `model/af3_teacher.py`、`model/r48k_geometry.py`、`model/sd_step.py`、`model/sdh_kernel.py`、`model/struct_distill.py`、`tests/test_f0b_structure.py`；
- 修改文件被限定为 `train_rqvae.py`；
- 几何四件 `model/mixed_rq.py`、`model/model.py`、`model/layers.py`、`model/poincare_contract.py` 与 F0 **SHA 相同**。
