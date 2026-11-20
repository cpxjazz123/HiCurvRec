## 结项：有向结构进入 RQ-VAE 的完整 Stage2 对照已完成，判定为负结果并停止

按 #221 §3–§6 要求的链路逐项交付，全部证据在 `results/research/directed_graphlet_role/`
（`MANIFEST.json` 逐文件 sha256；`CGV.md` 是 CGV6 的可执行证据索引；`ABLATION.md` 是消融表本体）。

### 1. 消融表（Stage2 冻结配方 3000 epoch、seed 42，取重跑值）

| arm | L1 | L2 | L3 | collision | subtype F1 | orbit F1 | test_R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| D0 (no aux) | 256 | 7964 | 20596 | 0.1623 | — | — | — |
| D1 (directed, head on z) | 256 | 10887 | 23964 | 0.0253 | 0.0774 | 0.5071 | 0.0444 |
| U (undirected labels) | 230 | 8565 | 23037 | 0.0630 | 0.7066 | 0.6480 | — |
| S (shuffled directions) | 247 | 9386 | 23640 | 0.0385 | 0.0709 | 0.4905 | — |
| frozen baseline | 249 | 7374 | 20425 | 0.1693 | — | — | 0.0555 |
| mixed F0 (reference) | 79 | 4151 | 18264 | 0.2572 | — | — | 0.0572 |

**本轮只看 Stage2**（owner 2026-11-20 指令）：U 与 S 不进入 Stage3，因此 §6 的 recall 判据本轮不作结论；全库唯一的 Stage3 证据仍是 D1 的 `test_recall@10 = 0.04438`（冻结基线 0.0555）。

### 2. 三条结论

1. **给定邻接，结构可识别**：subtype macro-F1 0.712、无条件 orbit 0.610；方向打乱后 0.169/0.146，邻接清零 0.030/0.012。
2. **不给邻接，结构不可预测**：z/q/SID/DGDV/度数/Dir-GNN 全部落在 0.493–0.514（SE 0.008）。行为边的存在性可读（分层负样本 AUC 0.69–0.73，度数基线 0.535），但**方向不可读**：四条臂、z/q/SID 六种组合的方向准确率 0.497–0.510，训练集方向损失停在 ln 2；λ=10 时方向项只能以压掉 AUC 到 0.59–0.66 为代价。
3. **结构监督进量化器没有兑现**：D1 的 SID 更细（碰撞 0.0253）但 SID→orbit 探针在任何深度都低于多数类基线，Stage3 0.0444 < 0.0555；且方向标签的行为边保留（L1 0.0105）**低于**无向标签（0.0200）与打乱方向（0.0122）——方向信息没有转成结构保真。

### 3. 本轮修掉的工程缺陷（每个都有回归测试）

- 无向/打乱参照臂从未跑通：词表缺 `allowed`、无向标签切错行、打乱把局部 orbit id 当全局、打乱破坏互惠对、训练后评估用错词表；
- 参照臂原先建在 `ARM_VARIANT=D3`（梯度进码本），epoch 49 触发码本健康门；改为 D1 后才成为「只差标签」的对照；
- DGDV 的 held-out 边按有序对过滤，反向边漏回计数图（修好后此前「有向 DGDV +0.026」归零）；
- 方向读出缺少方向目标、负样本是热度检测器（改为分层负样本后度数基线 0.852→0.535）；
- **Teacher 没有落盘**：臂 checkpoint 只含 encoder+quantiser（19 key），联合训练的头随进程消失。现由 `teacher.py` 提供保存/冻结/重载契约（7 条测试：logits 逐位一致、冻结后无 autograd 图且参数不动、词表/维度不符拒绝、裸 state_dict 拒绝、legacy 头无损迁移），trainer 在 heldout 评分后写 `logs/teacher.pt`。已迁移的 Teacher：`head_pretrained_directed.teacher.pt`（sha `8efcdc8d…`，源 sha `5de4b4bb…`，subtype 0.0206 / orbit 0.3440）。

### 4. 重复性（必须与任何单次数字一起引用）

同一配方同种子重跑**不逐位复现**（`CuBLAS ... not deterministic`）：U 碰撞 0.0763→0.0630、S 0.0337→0.0385，两次 `item_sids.json` sha 不同。单次 SID 指标自带约 ±0.005–0.013 的带宽。

### 5. 未执行的部分（明确记录）

- **802 维 DGDV 输入臂不启动**：离线闸门在修掉泄漏后为 0，按 §19g 的一次性预算不投入；
- **U/S 的 Stage3 不跑**（owner 指令）；D0 的 Stage3 按 §89 门槛本就不可能（碰撞 0.1623 > 0.10）。

复现命令见 `REPORT_DIRECTED_GRAPHLET_ROLE.md` 各节末尾、`CGV.md` 与 `post_stage2.sh`。
