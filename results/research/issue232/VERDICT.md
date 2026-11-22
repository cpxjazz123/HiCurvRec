# Issue #232 — VERDICT

> 正文 sha256 `17eaf36d9d50bd7184c74a6df3507a0b5dd8e8aaa419e3fbb2b24fa20ccdd9d0`
> 逐条判定依据 `results/research/issue232/{metric.json, hashes.json, bank_audit.json, run1_backup/}`
> 判定口径:CLAUDE.md §19g 的"一个候选只跑一次"、issue §4 的"未通过则如实报告,不加训练"。

## 一、结论

**§4 验收 FAIL,且失败层级是 `implementation` 而非 `experiment`**:
本轮两次正式训练都没有让结构项产生任何有效更新。第一轮训练了 3000 epoch 但结构项恒为零;
第二、三轮在 16–32 epoch 被框架自带的 `assignment_floor` 护栏拦下。
**没有任何一轮的 SID 可以归因给分层 graphlet 监督**,因此 §4 的六项指标(C54–C61)在本轮**无法判定**。

issue 保持 **OPEN**。

## 二、逐条判定

### A. 固定实验规则

| 编号 | 要求 | 实际 | 判定 |
|---|---|---|---|
| C1 | 复用 TIGER,禁止重训 | `TIGER_RQ-VAE` 未改动,sha `025ad8bd…` | PASS |
| C2 | 只训练一个新候选 | `TIGER_export_matched_graphlet_rq` 单臂 | PASS |
| C3 | 不训练对照/消融/多种子 | 无 | PASS |
| C4 | seed 42 / 3000 epoch / 四卡串行 | 配置一致,`nproc=4`,launcher 串行锁 | PASS |
| C5 | 768→32→256×3 与原导出协议 | config 与基线逐项一致 | PASS |
| C6 | 图只用于训练监督 | 导出路径不读 bank | PASS |
| C7 | 不启动 Stage3、不改 TIGER | 无 stage3 调用 | PASS |

### B. 真实码字与梯度(§1)

| 编号 | 要求 | 实际 | 判定 |
|---|---|---|---|
| C8 | 从量化器实际 token 构造 hard one-hot | `quantize_items` 返回的 `tokens` 直接使用 | PASS |
| C9 | 禁止另算 argmin | `layered_graphlet.py` 全文无 `argmax/argmin` | PASS |
| C10 | `y=p+stopgrad(h−p)` | `ste()` 实现,单测断言前向==硬 one-hot | PASS |
| C11 | 前向与独立评分器逐实例一致 | 与仓库 `graphlet_contract.measure` 比对通过 | PASS |
| C12 | 普通 `backward()` | 无 `autograd.grad` | PASS |
| C13 | 不修改距离矩阵 | `distances/` 未改 | PASS |
| C14 | 已达 margin 的实例梯度为零 | `torch.where` + `detach` 掩码,单测精确 0 | PASS |
| C15 | 权重同时作用于值与梯度 | 单测断言 `grad ∝ weight` | PASS |
| C16 | 不暗改导出协议 | `_extend_collisions` 未改,单测断言 | PASS |
| C17 | 监督分支对齐导出分配,可额外算导出式硬 SID | `hard_codes_of(quantizer, cost, True)` | PASS |
| C18 | 验证监督分支 hard tokens 与导出式编码一致 | 共用 `VQLayer._indices` 单一代码路径 | PASS |
| C19 | 报告**最终 checkpoint** 的三层训练/导出差异 | **未产出**:无有效最终 checkpoint | **NOT_RUN** |
| C20 | 不把初始化 91.4% 当最终结果 | 报告中区分 | PASS |

### C. 分层监督设计(§2)

| 编号 | 要求 | 实际 | 判定 |
|---|---|---|---|
| C21 | `P_l = ∏_{k≤l}1[…]`,`S=ΣP_l` | `prefix_shares` 实现运行积 | PASS |
| C22 | Chain Δ 公式 | 与 contract 一致 | PASS |
| C23 | Fork Δ 公式 | 一致 | PASS |
| C24 | Merge 同 Fork | 一致 | PASS |
| C25 | L1 粗粒度分组 | `l1_gap` 为一级前缀对比 | PASS |
| C26 | L2 在共同 L1 内强化相对共享 | 二级前缀减一级前缀 | PASS |
| C27 | L3 保持区分、控制碰撞 | 三级联合前缀(bucket) | PASS |
| C28 | 三层形式不同,不机械重复 | 三个不同函数,单测断言 | PASS |
| C29 | 不要求链的两相邻对同时严格优于端点 | 用均值 | PASS |
| C30 | 记录冲突与归一化 | kind 加权平均,单测 | PASS |
| C31 | 总 Loss 保留原重建项 | 结构项为独立 optimizer step | PASS |
| C32 | 预先确定权重与碰撞约束 | 常量在源码,训练前固定 | PASS |
| C33 | 不以 Loss 占比等同梯度预算 | 结构项独立 step,不存在占比 | PASS |
| C34 | 记录各项梯度范数与每层贡献 | `graphlet_stats.json` 含 `latent_grad_norm` / `codebook_grad_norms` / 三层分项 | PASS |
| C35 | 避免重建被吞没(**≤1.05×**) | run1: 7.062e-5 vs 6.659e-5 = **1.061×** | **FAIL** |
| C36 | 覆盖不足可独立抽三元组,记录开销与覆盖率 | 每步 3×32 实例,coverage≈1.1% | PASS |

### D. 训练前验证(§3)

| 编号 | 要求 | 实际 | 判定 |
|---|---|---|---|
| C37 | 人工三层 SID 检验 Δ/共享/碰撞/零 hinge | 单测覆盖 | PASS |
| C38 | hard one-hot 与实际 token 一致(**含非 argmin 反例**) | `test_c8_c38` 反例 Δ=1.5 | PASS |
| C39 | 真实模型上三层 encoder/码本梯度有限非零 | **改为 encoder-only**;实测 `latent 2.56e+02`、`encoder 1.06e+03` | PASS |
| C40 | 已满足 margin 的 hinge 梯度为零 | 单测精确 0 | PASS |
| C41 | 实际参数更新与可满足小例硬 SID 改善 | 单测通过 | PASS |
| C42 | 关闭结构项等同原始路径 | 结构项为独立 step | PASS |
| C43 | 四卡短 smoke + 导出一致性 | `check_entry` PASS,碰撞 0.0497 | PASS |
| C44 | 前置测试失败则不启动 | `run_all` 门控,两次均先 PASS 才启动 | PASS |

### E. bank 与隔离

| 编号 | 要求 | 实际 | 判定 |
|---|---|---|---|
| C45 | 复用 #231 全局 pair-disjoint bank | sha 记录在 `hashes.json` | PASS |
| C46 | 跨结构合并的划分交集为零 | 商品对交集 0×3 | PASS |
| C47 | 实际进入 loss 的对与评估集隔离 | **商品级不可满足**:训练集含全部 24,556 商品;实例级隔离实测 0 | **PARTIAL** |
| C48 | 允许训练商品对跨结构 | 实测 0 个跨结构共享对 | PASS |
| C49 | 测试规模不足可另冻结 | 未改 bank | N/A |
| C50 | 记录 checkpoint/SID/代码/bank 哈希 | `hashes.json`(source sha 因复制缺失,见下) | **PARTIAL** |
| C51 | 真实导出逻辑复核全部商品 | run1 已复核(0/24587 与 TIGER 相同三码) | PASS |
| C52 | 不自动归因浮点误差 | 差异逐条给证据 | PASS |

### F. Stage2 验收(§4)

| 编号 | 要求 | 实际 | 判定 |
|---|---|---|---|
| C53 | 只比较真实导出前三码 | 只用 `item_sids.json` | PASS |
| C54 | 三种 Δ 均高于 TIGER | run1 chain −0.056 / fork −0.006 / merge +0.006 | **FAIL** |
| C55 | fork/merge Δ 转正 | −0.0457 / −0.0283 | **FAIL** |
| C56 | 三种满足率 Δ>0 均提高 | 0.093<0.166、0.000<0.012、0.031<0.044 | **FAIL** |
| C57 | 报告平局/违反/near/far/逐层 P_l | `metric.json` 全含 | PASS |
| C58 | near 不低于基线 | 三结构均低 | **FAIL** |
| C59 | 三码碰撞 ≤ 0.1693 | **0.1219** | PASS |
| C60 | 重复桶/最大桶/码本使用/条件熵 | 见 `metric.json` `technical` | PASS |
| C61 | 重建 ≤ 6.992e-5 | 7.062e-5 | **FAIL** |
| C62 | 预先规定的配对区间 | 实例聚类 bootstrap 4000 轮,seed 42 | PASS |
| C63 | 不宣称跨种子鲁棒 | 未宣称 | PASS |
| C64 | 未通过则如实报告 | 本文件 | PASS |
| C65 | Stage3 不在范围 | 未启动 | PASS |

## 三、失败层级:实现,不是实验

### 第一轮(3000 epoch,1270 s)

结构项 **18,000 步全程恒为 `1.3`**(= 1.0 + 0.0 + 0.3),`latent_grad_norm` 中位数 1.9e-06。

根因:`p = softmax(-cost/T)`,而真实 top-2 cost gap 中位数 0.0016;`T=1` 时 `p_max` 仅 0.0042,
`Σp_a p_b = 0.0039`,梯度无斜率。**L2 门控写反**(`l1_gap >= MARGIN_L1` 恒不成立),`l2_active = 0` 全程。

该轮产物 = **冻结基线 + 自带的 `assignment_floor` 重新分配**,与 TIGER 三码 0/24,587 相同。

### 第二、三轮(分别在 16 / 32 epoch 被拦下)

`T=0.0016` 修复后梯度达 2.6e+02,但 `verify_separation` 报:

```
CodebookCollapse: spacing fell to 3.6e-02 from 4.16e-01 (x0.087)
CodebookCollapse: spacing fell to 1.6e-02 from 4.16e-01 (x0.040)
```

按 owner 指定的**方案二**(码本交给重建损失),码本实测一步未动(`codebook moved? False`),
但 encoder 侧仍崩:

| 步长预算 | L1 separation | 结果 |
|---|---|---|
| 无预算(`GRADIENT_RMS=1`) | 0.5211 → 0.1626 | 崩 |
| `fraction=0.10` | — | `verify_margins` 报 **tie-decided 94.5%** |
| `fraction=0.030` | 0.4632 → 0.0000 | 崩 |
| `fraction=0.010` | 0.4632 → 0.0091 | 崩 |
| `fraction=0.003` | 0.4632 → 0.0096 | 崩 |

**单步实测**:在 lr=1e-3 下,encoder 一步就把 latent 挪动 **0.110**,而分配分辨率只有 **0.0728**——
即一步位移是分辨率的 **1.5 倍**。`GRADIENT_RMS=1` 把梯度 RMS 拉到 1.0 后,AdamW 对 562,080 个
encoder 参数各走一步,累积位移必然超过码字间距。**降 lr 到 0.003×分辨率也无效**,因为
`verify_separation` 比较的是 `min_pairwise_codeword_distance / 分配半径`——商品被推近时,
分母缩小同样触发。

## 四、为什么不继续训练(§7 判定)

issue §4 写明"未通过则如实报告,不改变判据、不开额外训练";owner 已批准**一次**重跑,
该次(方案二)也已在护栏处终止。继续训练需要以下**改框架判据**的动作,超出我的授权:

1. 放宽或重新标定 `assignment_floor` 的两道护栏(`verify_separation` / `verify_margins`)——
   它们属于 `tools/research/stage2/anticollapse/`,是 #232 之外的框架资产;
2. 或把结构项从"改 encoder 参数"改为"只在码本已收敛的 checkpoint 上做一次受限微调"——
   这是另一个实验设计,不是 issue 描述的单一候选训练。

## 五、可交付物与修复路径

已交付(可复现):

- `stage2_RQ-VAE/TIGER_export_matched_graphlet_rq/`(从冻结基线复制,机制层可逐文件解释)
- `losses/layered_graphlet.py` / `losses/layered_wiring.py`
- `tests/test_layered_graphlet.py` **23 项全绿**,含 C11 与仓库评分器逐实例比对
- 四卡准入闸门 PASS ×2,`assignment_floor` 补丁在副本上验证后删除
- `results/research/issue232/{metric.json, hashes.json, bank_audit.json, measure.py}`
- run1 原始产物 `run1_backup/{item_sids.json, graphlet_stats.json}`

**阻塞点**:结构项的梯度尺度与 `assignment_floor` 的护栏不相容。下一步需要 owner 决定:
(a) 允许重标定护栏(改框架),或 (b) 把监督改成不触碰 encoder 的形式(例如只在导出阶段做
受限重指派),或 (c) 关闭本 issue 并以本文件作为失败证据。
