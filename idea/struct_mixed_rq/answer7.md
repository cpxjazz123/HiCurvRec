# answer7 — 采集器修正、状态区分与独立 H 参考；`answer6` 的潜变量数值已换到可追溯出处

**状态：CLOSED（测量与数学对照完成；规格/前缀/准入/训练/收益均未运行）**　完成：2026-11-10　对应 `question7.md`

---

## 1. 撤回与更正

| 项 | 处置 |
|---|---|
| `answer6` §4 的结构潜变量 `3.8047e-05 / 4.2535e-05` | **撤回原出处**：`q6/factors.json` 当时记录为 `structure: NOT_CONNECTED`，该文件哈希 `b1675fb1…` 未变（本轮用 `git checkout` 恢复，未覆盖）。同一对数值现由 `q7/factors.json` 复现，见 §4；`answer6` 原文保留供追溯 |
| 采集器把"没找到张量"标成"未连接" | **确认是缺陷**：`latent_grad` 用请求数 `3*TRIPLES=192` 过滤，而实际接受 45 个三元组只有 `3×45=135` 行，找不到即返回 `NOT_CONNECTED`。已修，见 §2 |
| 参数块把 `None` 与"连接但为零"合并 | **确认是缺陷**：拼接前对 `None` 填零，诊断状态丢失。已修，见 §3 |

---

## 2. 潜变量采集：真实红灯 → 修 → 绿

测试 `tests/test_q7_latent_capture.py`（真实模型、真实 checkpoint）：

```
AssertionError: expected a latent gradient tensor; got 'NOT_CONNECTED'
(the capture filtered by requested triples, not accepted ones)
```

原因是行数筛选用 `3*TRIPLES`。修复：

```diff
-        rows = ... 3 * TRIPLES                      # 192，但只接受了 45 个三元组
+        rows = ... 3 * len(triples)                 # 135
         chosen = [z for z in captured if z.shape[0] == rows]
         if not chosen:
-            return "NOT_CONNECTED"
+            return "NOT_MEASURED"                   # 采集缺口，不是物理未连接
         z = chosen[0]
-        return z.grad... if z.grad is not None else None
+        return "NOT_CONNECTED" if z.grad is None else z.grad.detach().clone()
```

修后该测试通过；结构潜变量在 `q7/factors.json` 中实际记录为
`structure: {"split":"E16/H16","E_norm":3.804699513201433e-05,"H_norm":4.253528175652463e-05}`。

`recon`/`quant`/`total` 仍为 `NOT_MEASURED`（它们与结构项使用不同商品集合，`total` 无法对应单一潜变量张量），不作物理结论。

---

## 3. 状态区分与 checkpoint 回归

新增 `probe_factors.block_status(params, grads)`：分别统计 `none_tensors`、`zero_tensors`、`nonzero_tensors` 与范数，不再用填零掩盖未连接。

`tests/test_q7_states_and_checkpoint.py`（通过）：

- 一个 `None` + 一个全零梯度 → `none=1, zero=1, nonzero=0`；
- 真实 checkpoint：`load_state_dict` 无 missing/unexpected，且模型权重与文件张量 `torch.equal`（证明真正恢复，而非仅调用 API）；
- 截断的 state 在 `strict=True` 下抛 `RuntimeError`（不兼容状态被拒绝）。

---

## 4. q7 真实 TRAIN 运行（修复树，旧 checkpoint，标注为"旧 checkpoint / 新接线"）

身份：`ARM_NAME=TIGER_struct_mixed_rq_ste_fix`，checkpoint SHA `84f05d46…4672`，SID SHA `7c147fc6…90a1`，TRAIN 前 64 个商品（0–63），三元组接受 **45/64**，`s=0.04444597661495209`、`λ=9.226019983543665e-07`（旧运行标定，非修复版正式标定），mode=train，`behaviour_on=True`、`geometry_on=True`。

| 项 | 值 |
|---|---|
| recon / quant / λ·structure / total | `1.0601e-04` / `1.6660e-04` / `3.96e-06` / `2.7657e-04` |
| encoder（整块，`NOT_APPLICABLE`） | `4.7198e-05` |
| C1 E / H | `3.0060e-05` / `3.0355e-05` |
| C2 E / H | `3.9891e-05` / `4.4939e-05` |
| C3 E / H | `3.8190e-05` / `4.2658e-05` |
| decoder | `NOT_APPLICABLE`，范数 0 |
| 结构潜变量 E / H | `3.8047e-05` / `4.2535e-05` |
| `base + λ·structure` 与 total 逐张量最大偏差 | `2.9103830456733704e-11` |

这些数值逐项来自 `results/research/struct_mixed_rq/q7/factors.json`（SHA 见 §7），未拼接旧版本。

---

## 5. H 原生累计回读的独立参考

`tests/test_q7_h_reference.py`（通过）：测试内独立实现
`ball(h)=h·tanh‖h‖/‖h‖`、`log0(y)=y·atanh‖y‖/‖y‖`、Möbius 加法与右嵌套 `c1⊕(c2⊕c3)`，
对 1/2/3 层累计回读与 `native_product.reconstruct` 比较：

- 取值：`rtol 1e-10 / atol 1e-12` 全部通过（非共线 H、球内安全）；
- 对三个选中码字的梯度：`rtol 1e-8 / atol 1e-10` 全部通过。

该测试不调用被测实现生成期望值；离散 argmin 与 STE 代理不在其中（由 §2 的固定选码测试覆盖）。

---

## 6. 规格与身份缺口（明确保留）

**`SPECIFICATION_INCOMPLETE`**（原 idea §4.1 三层前缀监督需要的定义，未擅自补）：

1. 三个累计位置 `c1`、`c1⊕c2`、`c1⊕(c2⊕c3)` 的结构项如何聚合（求和/平均/权重？原 idea 未定义，且禁止按层调权）；
2. 共用 `s` 在哪个表面、按什么规则标定（末层？逐层中位？）；
3. `λ` 的块预算针对哪个"完整结构目标"（三层之和？每层各自预算取 min？）。

**未运行项**：

| 项 | 状态 |
|---|---|
| 数学规格冻结 | `NOT_RUN`（`math_spec.json` 仍 DRAFT，含旧 `ratio=0.1` 与累计码字求和描述） |
| 前缀监督实现 | `NOT_IMPLEMENTED`（等待上述定义） |
| 机器准入（smoke/DDP） | `NOT_RUN` |
| 正式训练（修复版） | `NOT_RUN` |
| 下游收益 | `NOT_RUN`；旧 `test_R@10=0.05790490781524748` 属旧源码图，不继承 |
| 实际运行绑定 | 部分：`ARM_NAME` 已分离，但 `LOG_DIR` 仍指向源码树 `logs/`；框架的不可变快照与独立结果目录绑定 `NOT_RUN` |

---

## 7. 本轮执行与身份

| 项 | 值 |
|---|---|
| 测试执行 | `files pass=21 fail=0`（逐文件 `rc` 与输出见 `q7/tests.log`） |
| 测量脚本 | `probe_q6_factors.py`（写入 q7）SHA `3808cc90…`；`probe_factors.py` SHA `cf002094…` |
| 修复树 `layers.py` / `train_rqvae.py` | `18e5d620…` / `db6ee335…` |
| 新增测试 | `test_q7_latent_capture.py` `13ae0187…`、`test_q7_states_and_checkpoint.py` `f9a3af89…`、`test_q7_h_reference.py` `276ad66b…` |
| 结果 | `q7/factors.json` `bfb8253c…`、`q7/factors_run.log` `b25cced3…`、`q7/tests.log` `e37dabbb…` |
| 未覆盖 | `q5/`、`q6/` 目录内容保持原样（q6 JSON 已恢复为提交版本） |

---

## 8. 结论格式（逐项）

| 项 | 状态 |
|---|---|
| 测量修复（采集/状态/checkpoint） | **DONE**，含真实红灯→绿与回归 |
| 当前本地测试 | **DONE**：21 文件 / 0 失败 |
| H 数学对照 | **DONE**：取值与梯度独立参考一致 |
| 规格冻结 | **NOT_RUN** |
| 前缀实现 | **NOT_IMPLEMENTED**（`SPECIFICATION_INCOMPLETE` 三项） |
| 机器准入 | **NOT_RUN** |
| 正式训练 | **NOT_RUN** |
| 真实收益 | **NOT_RUN**（旧 0.05790 不继承） |

本问不关闭机制主线，也不创建新 idea；修复版无 Stage2/Stage3 成绩。
