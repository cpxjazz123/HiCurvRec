# Issue #230 — 预处理证据与搁置理由

> 状态：**按 skill §3 判定为"契约缺陷未修完"，issue 保持 OPEN**。
> 本轮只完成 §1–§2 的评估合同(代码 + 单测),未启动任何训练。

## 本轮已完成(有证据)

| 项 | 产物 | 证据 |
|---|---|---|
| 评估合同单一实现 | `analysis_sid_recall/graphlet_contract.py` | 13 个单测全绿 |
| 角色顺序统一(A=中心;chain A→B→C) | 同上,`_REQUIRED` / `_ORDER_SRC` / `_ORDER_DST` | 采样与评分共用同一张表 |
| 严格诱导(六条有向位置全检查) | `induced_mask()` | 四个额外位置逐个单测覆盖 |
| 修正 merge 的 `(B,C,A)` 与评分器不一致 | 采样器对三种结构都输出 `(A,B,C)` | `test_fork_and_merge_use_the_centre_at_position_zero` 用旧 tuple 断言必须不合格 |
| 修正 chain 漏检 B→A | `induced_mask` 检查 `BA` | `test_chain_requires_both_forward_edges_only` 四次注入全部判假 |
| Δ 口径(链用相邻对均值,不要求两条严格) | `delta_of()` | `(3,1,2)→0`、`(2,3,1)→1.5` 两个人手算例 |
| S() 只用前三码 | `shared_prefix()` | 第 4 列不同仍记 3 |
| 平局/违反/满足率与 near/far 绝对共享 | `measure()` | 三率之和 == 1 的单测 |
| 真实有向图上的诱导率 | `results/research/issue230/induced_audit.json` | 三结构各 20000 条全部严格诱导 |

真实图规模:`results/research/behavior_graph_directed/graph.npz`,24,587 节点 / 323,633 条有向边,
来源 `train.parquet`(sha256 `80597e38…`),`validation_test: not read`。

## 未完成项(因此 issue 不能关闭)

| 编号 | issue 要求 | 现状 |
|---|---|---|
| C3(完整) | 同步修复 `evaluate_arm`、`arm_banks`、`sid_alignment_v2` 与**训练侧** | 三个分析脚本仍各自持有采样器/评分器;**训练树 `stage2_RQ-VAE/TIGER_graphlet_wide_l1_rq/losses/chain_align.py` 仍是旧角色定义**。只统一了新建的合同模块,旧实现未切换 |
| C6 | 监督只读训练划分 | 未写运行时断言 |
| C7 | graphlet train/validation/test banks 无序对不重叠 | 未生成三向切分,未做集合断言 |
| C8 | 冲突样本丢弃并记录 | 未实现 |
| C9 | 训练期实际 pair(含负样本)与评估 bank 交集断言 | 未实现 |
| C11 | 固定抽样与度数分布记录、bank sha256 | 只有 sample 前三条,无度数分布与 sha256 落盘 |
| C12 | 从 checkpoint 重编码全部 24,587 商品并逐项核对 SID | 未执行 |
| C14 | 用新合同重算基线与历史模型 Δ | 未执行 |
| C22 | 训练 graphlet 冲突率(同对被拉近/推远)分析 | 未执行 |
| C33 | 四组对照(基线 / 旧 soft loss 核验 / 新高前缀监督 / 打乱对照) | 未实现新机制,未核验旧臂可复用性 |
| C36–C47 | §5 成功判据全部 | 需要训练,未启动 |

## 为什么停在这里(§3 的判定)

skill §3 要求"全部 PASS 且没有解释性放宽"才允许进入 §4。当前 C3 仍然 FAIL:issue 明确写
"同步修复训练、`evaluate_arm`、`arm_banks`、`sid_alignment_v2` 等相关路径",而本轮只建立了新合同模块,
旧路径仍在运行。在旧角色定义还在训练侧生效的情况下开跑 3000 epoch,产出的 Δ 无法归因,等于重犯 §1 点名的错误。

## 下一步(按依赖顺序,不需要新授权)

1. 把训练树与三个分析脚本切到 `graphlet_contract`,旧实现删除而不是并存(C3 完成)。
2. 生成 train/validation/test 三套 banks + 无序对交集断言 + 冲突丢弃记录 + 度数分布 + sha256(C6–C11)。
3. 冲突率与不可满足约束分析(C22),一条命令产出报告。
4. checkpoint 重编码核对脚本(C12/C13),同时重算基线 Δ(C14)。
5. 只有 1–4 全绿,才按 §4 写新机制的 TDD 测试,再进入两个强制训练臂(新机制 + 打乱对照)。
