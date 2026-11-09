# answer13 — 前缀候选进入 DRAFT、诊断迁移完成；冻结仍阻塞、前缀未实现

**状态：CLOSED（§2/§3/§4 本地交付完成）；冻结 BLOCKED，前缀与收益 NOT_RUN**　完成：2026-11-10　对应 `question13.md`

---

## 1. §2 前缀候选写入现有 DRAFT（DONE）

`stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/math_spec.json` 实际修改（SHA `fc2fb26c…` → **`f254684f…`**），`status` 仍为 `DRAFT — revised per review; not admitted, not frozen`：

| 字段 | 改动 |
|---|---|
| `outputs.positions` | 由 "cumulative codeword sum" 改为**右嵌套回读经 log0**（E 求和 + `c1⊕(c2⊕c3)`，decoder 消费 log0 回读） |
| `parameters` | **删除 `ratio: 0.1`**；`weight_rule` 改为对完整前缀目标的一次两块梯度预算（encoder + 全部码本；decoder 不入最小值；无逐层权重、无 ratio、无扫描） |
| `computations` | 写入：H→球映射（仅饱和尾巴回拉）、`||e1−e2||²+φ(t)`、**一次真实选码产生三个累计位置**（不重复 quantize/gather）、逐层 `L_l`、`L_prefix=(L_1+L_2+L_3)/3`、共享尺度（末层初始化、`sqrt(clamp_min(d²,1e-8))` 中位数、空样本退化） |
| `gradient_rules` | 每层**单位 STE** `q^(l)=R_l+z−sg(z)`；码本只走原生回读连续 Jacobian（含前层码字间接通路）；量化损失独立 gather |
| `training_integration.prefix_objective` | `L=L_base+λL_prefix`，λ 在 optimizer step 前一次标定；`λ_mean=3λ_sum` 的适用前提与"不等价于末层监督" |
| `switches.prefix` | behaviour-off 移除结构项；geometry-off 同时切换选码/残差/回读；both-on 唯一训练臂 |
| `excluded` | 追加硬选码不可微、STE 不以有限差分证明 |

未把 DRAFT 改为 FROZEN，未伪造 reviewer/推导身份。

---

## 2. §3 诊断迁移（DONE）

1. **单一活动实现**：`probe_q6_factors.py` 改为显式退役（导入即 `ImportError`，指向 `probe_q9`；历史 q6/q7 报告保留在 results 下）；`test_q8_report.py`、`test_q7_states_and_checkpoint.py` 改指向 `probe_q9`。
2. **真实身份**：`probe_q9` 报告写入**有序三元组 `[centre, a, b, type]`、`triples_sha256`、TRAIN/embedding/checkpoint/SID 的 SHA、6 个候选源码文件 SHA**；`z_struct_rows` 由同一份三元组决定（本轮实测 64 请求 / 45 接受 / 135 行）。
3. **同数量异样本反例**：`tests/test_q13_identity.py::test_same_count_different_items_is_rejected`——仅替换一个三元组，身份检测器即拒绝。
4. **共享语义**：`training_hooks.structure_term` 新增可选 `triples=`，诊断与训练走同一个结构损失入口（不再自编排采样）；行为开关与空样本处理沿用该函数。
5. **历史证据不写入**：测试使用 `tempfile`；诊断输出改为 `run-<timestamp>` **不可覆盖目录**（连续两次运行产生两个独立目录，测试断言）。
6. **checkpoint 拒绝经消费者**：截断 state 由 `probe_q9.run` 抛 `SystemExit`，且不产生 `run-*/factors.json`（测试断言）。

---

## 3. §4 收尾（部分）

- 负对照现在调用与正对照**相同的** `assert_matches`，并在同一次 `code_ball` 替换内断言 E 梯度非零、`d(level)/dz` 仍为 32×32 单位阵；
- 拒绝记录：`(1,0,1.187)`、`(1,1,0.970)`、`(2,0,1.821)`、`(2,1,1.131)`、`(2,2,0.881)`；
- **诚实缺口**：`k=0` 的一对未被比较器拒绝（5/6 组），原因未定位，保留为待查；
- 已删除 `test_q10` 中对参考叶子求 `bad` 梯度的空洞断言，保留有效对照。

---

## 4. 状态

| 项 | 状态 |
|---|---|
| 独立审查/冻结 | **BLOCKED**（缺失前提：符合当前规则、可用且有独立执行身份的审查入口；不恢复已删除 DAG/Agent 调度器） |
| 前缀训练实现 | `NOT_IMPLEMENTED`（未冻结前不做） |
| 框架运行绑定（`LOG_DIR` 仍在源码树） | `NOT_RUN` |
| 机器准入 / 正式训练 / 下游收益 | `NOT_RUN`；旧 `0.05790490781524748` 不继承 |

测试：`files pass=25 fail=0`（`q13/tests.log`；该文件顶部仍残留一次中断运行的旧汇总行，最终汇总行为 25/0）。

---

## 5. 身份

| 项 | 路径 | SHA-256（前 20 位） |
|---|---|---|
| DRAFT 规格 | `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/math_spec.json` | `f254684f9a0e4b0c4a20` |
| 活动生产者 | 同树 `probe_q9.py` | `933573cac55df2391253` |
| 结构入口改动 | 同树 `training_hooks.py` | `8226380f4c8c355ff30d` |
| 迁移/身份测试 | `tests/test_q13_identity.py` | `7123c754fc5285c8f2fe` |
| 反例测试 | `tests/test_q12_h_breaker.py` | `23963c088999c89fbd01` |
| 原 Jacobian 测试 | `tests/test_q10_native_and_rqlayer.py` | `b5132e9be148f1f03c4b` |
| 套件日志 | `results/research/struct_mixed_rq/q13/tests.log` | `8d4e3d78718e36fc1d6f` |

q5–q12 历史产物未覆盖；不派发正式训练、不覆盖原实验与 R2 编号 57；不新建 idea。
