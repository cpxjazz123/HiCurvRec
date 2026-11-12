## 完成：三级曲率在**首个计划评估点**触发塌缩闸门，按预注册记负结果收线

产物：`results/research/issue207/`（`FINDINGS.md`、`CLOSURE.md`）。

### 冒烟（第 2 次派发通过）

`SMOKE_PASS`：`steps = 24`、8 项梯度范数全正（0.0007–0.0026）、`dead_terms = []`、`nonfinite_terms = []`；冒烟日志确认作用层 **L1 False / L2 False / L3 True**、`level12_on=False`、κ 表 sha `270b34f2…`。
（第 1 次冒烟触发 `collision 0.846667 at epoch 1` 属**冒烟伪影**——一 epoch 冒烟在码本刚初始化、只走一步时评估；闸门已限定为 `EPOCHS >= EVAL_INTERVAL`，只判断真实训练。）

### 真实运行

| 项 | 值 |
|---|---|
| epochs 1–49 | 正常（epoch 49：`loss 1.598e-04`、`lr 9.833e-04`） |
| **首个计划评估（epoch 50）** | **`collision = 0.5044129011266116`** |
| 闸门 | `COLLAPSE_GATE_TRIGGERED` ⇒ `SystemExit`，**未导出 SID** |
| Stage3 | **未启动**；`test_R@10 = NOT_RUN` |

### 如实标注两点

1. **"3000 epoch 会不会自我修复"是未测**（闸门按预注册禁止继续）——能写的只有"首个计划评估点已 ≥ 0.40"。
2. 允许类"三级"这一格记为 **gated（被闸门终止）**，**不是**测得的负值。四层位（L1 的 LKC/CMR/MBR/SIGN、L2 的 L2K、三层的 HC-3、三级的本臂）与损失侧（F0V）**没有任何一格给出正的 `test_R@10`**。
3. 与 #206 并列：损失侧在 epoch 1 塌缩、纯几何的三级在 epoch 50 塌缩——是**同一类 SID 空间收缩**在两个入口出现，成因仍 `NOT_IDENTIFIED`。

### 验收对照（7 项）

1. `BEHAVIOUR_OFF` 全 κ=1、三层全 False；2. **一二级码分配在 `LEVEL12_ON` 前后逐值相同**（机器断言）；3. `d_κ` 与冻结合同 rtol 1e-6、重合点 <1e-5、边界与 κ=0.001 有限、gradcheck、编码器梯度；4. `SMOKE_PASS`；5. 闸门在真实 epoch 50 触发、谓词用例覆盖 0.9/0.5122/0.8234 触发与 0.3999/0.257168 不触发；6. §63/§6b **不适用**（闸门前终止）；7. 13 项用例全过，其中"几何未动"分两部分断言（`layers.py`/`poincare_contract.py` 与 F0 SHA 相同；`mixed_rq.py` 内 F0 的 `mixed_cost` 函数体逐字不变、新增定义只有 `mixed_cost_with_kappa`）。

### 裁决

塌缩闸门触发 ⇒ **立即停止、记负结果**；本线以"允许类零增益（上限 +1.9%）"结案，不重跑、不调参、不做消融。

**依据 commit**：`63ea8715b` · `c108ab8b2` · `da64e6d32` · `243fea767`。
