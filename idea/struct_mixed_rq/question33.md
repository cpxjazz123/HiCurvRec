# question33 — 目标改为"跑完一次完整 Stage2"：先补契约绑定，再过机器准入，再真训练

**状态：CLOSED（已审阅 answer33；未达新验收标准——无 Stage2，转 question34）**　承接：`answer32.md`。审阅结论与后续要求见 [`question34.md`](question34.md)。

## 0. 本轮目标与验收（硬）

**本轮的目标不是再修一处文档或测试，而是让 `TIGER_struct_mixed_rq_ste_fix` 真正跑完一次完整的 Stage2 RQ-VAE。**

**answer33 的验收标准 = 一次完整 Stage2**，缺任一条件即视为未答复：

1. 由 `stage2_start(source_dir, verification_id)` 启动，且该 `verification_id` 对应的 `verification_check` 是**真实执行**过四套独立测试并通过（不是引用旧收据）；
2. 训练配置为冻结配方：`world_size=4`、**全局 batch 4096**、`EPOCHS=3000`、`SEED=42`、无早停；**不得**为了并发把候选钉在单卡上；
3. 训练**跑完 3000 epoch**（不是中途停、不是只跑 smoke、不是 `epoch<N` 就结束），进程正常退出；
4. 产出实际文件：`results/stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/` 下的 `rqvae_best.pth`、`item_sids.json`、`logs/` 里逐 epoch 的 `training_metrics.jsonl`（含 `codeword_h_max`）、以及 `stage2_metrics/<sid_sha256>.json` 分层记录；
5. 报告里给出：run_id、实际 pid、起止时间与总墙钟、最后一行 epoch 记录（epoch=3000 的 loss）、`item_sids.json` 的 sha256、SID 的 L1/L2/L3 使用率与 L3 碰撞（诊断值，按 §19 不作门槛）。

只交代码、只交测试、只交设计说明，**都不算答复**——没有一次真实 Stage2 就是 `NOT_RUN`。

达到上述验收之后，**下一轮才谈 Stage3**；本轮不启动 Stage3。

## 1. 先决条件一：把 `mixed_distance` 绑到 16 维 `mixed_sq`

这是通过 `check_reference_agreement` 的最小一步，也是唯一能让"申报＝训练"成立的第一块：

- 新 call `mixed_distance_16`：`args` = `left`(width 32)、`right`(width 32)，`callable: mixed_geometry:mixed_sq`；
- 语句（可脚本生成）：`E = Σ_{i<16}(l_i−r_i)²`；每侧 `r=sqrt(Σ_{i≥16}t_i²)`、`b_i = tanh(r)/r·t_i`；`t = |b_l−b_r|²/((1−|b_l|²)(1−|b_r|²))`（注意方向，不是倒数）；`d = E + phi(t)`（`phi` 的两支如何写进语句必须显式，并与 `TAU` 一致）；
- 验收：`build_reference` 与 `mixed_sq` 在 64 组随机对（`‖H‖≤0.9`）、近碰撞（`‖ΔH‖=1e-6`）、边界（`‖H‖=0.9`）上取值 `rtol=0, atol=1e-12`、梯度 `rtol=0, atol=1e-10` 一致，两侧 `gradcheck`；`geometry_on=0` 单独一条断言；
- 若两支无法表达：允许只写渐近支并把该 call 的域限定为 `t ≥ TAU`，同时在 `coverage_gaps` 写明 `t<TAU` 无符号绑定。**不允许**再用 2 坐标面冒充 16 维。

## 2. 先决条件二：全维三层前缀面（288 标量）

`calls` 里仍没有任何 call 绑定 `reference_prefix_native`（三层累计回读 + `mean(L1,L2,L3)`）。做法：

- 生成器放进仓库（按 §86 只能放仓库内），例如 `results/research/struct_mixed_rq/q33/sympy_surface.py`，由 `run_suite.sh` 覆盖其 sha256；
- 生成的 `statements` 必须能被 `tools/research/stage2/reference.py::build_reference` 重建；
- 验收：与 `structure.reference_prefix_native` 在 `‖·‖∈{0.16, 0.9}` 与近碰撞点上取值 `atol=1e-12`、梯度 `atol=1e-10` 一致，两侧 `gradcheck`。

若这一步判断在合理时间内做不到，**当场说明并给出降级方案**（例如把机制降到最后层），交由 owner 决策；**不允许**再以"下一轮再做"收尾。

## 3. 冻结与启动（本轮必须走到启动）

`math_spec`/`reference_contract` 冻结 → `register_implementation` → `round_accept` → `verification_check`（真实跑四套测试）→ `stage2_start(source_dir, verification_id)`。按 §19g 只跑一次正式训练：不做开关消融、不做调参扫描、不做多种子；行为关/几何关开关只要求代码中存在并有廉价接线检查。

按 §59 常设授权，GPU 训练不需再逐次请示；但机器闸门不得绕过：任何一项 `FAIL`/`NOT_RUN` 都不许启动。

## 4. 监控与报告

监督期间单次等待不超过 60 秒，逐次报告已完成的 epoch、产物是否已生成、进程是否存活；用 `stage2_status` 或直接读日志/产物，不做长时间盲等。

## 5. 说明

上轮我错误地说"`CLAUDE.md` 没有第 86 条"，实际它在 `CLAUDE.md:206`（section 16）；`q31/DELETION_RECORD.md` 的处理是对的（日志本身不是 `/tmp` 文件，删它是错的，已记录）。此更正一并留档。

## 6. 审阅基线（本轮起始状态）

以下只绑定本轮开始时的候选状态，不替代独立审查、冻结或准入。

| 文件 | SHA-256 |
|---|---|
| `idea/struct_mixed_rq/answer32.md` | `942181d7c972154a1f5a1ee354a201ee033e31045e02d0ca5bb53d9cb9b7687e` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/math_spec.json` | `7e1f233790e3de44c9b6a9aeb295abdc2de0dd6e4f06ff4e8ed42e01f4dd8d4d` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/reference_contract.json` | `452487247825309cbeed50853f5c765774eed6d507c3baf170cac09ee850b312` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/structure.py` | `2e2fff8fc730d9cdca712cd7f469bb3972c5aaef09ca0803d55c12d68093040d` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/training_hooks.py` | `8ede9c091da752d02346ec7acb9fce25450aa14c3aa05d0cd274c1d2443beeff` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/train_rqvae.py` | `61dd1d9ada2b02aa752b9832951275819186c7e55d7d6781a761589d6160add9` |
| `tools/research/stage2/reference.py` | `176edf1191f65e603e8dffe79e815eaca732897c6c2683f9eced315e91105181` |

现状态：`results/stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/` **不存在**——这个候选一次都没训练过。
