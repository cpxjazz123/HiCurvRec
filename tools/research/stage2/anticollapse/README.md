# anticollapse — 让任意 Stage2 机制都无法饿死码字

## 一句话

把量化器的**分配**这一步换成"每个码字至少拿到 `floor` 个商品的最近邻分配"。损失、距离、曲率、码本初始化、编码器、全部超参**一律不动**,因此它对任何机制都成立,而不是对某一个机制成立。

## 为什么这是保证而不是惩罚

第 1 轮实测:码本项 `codebook_term` 用 `residual.detach()` 构造,**未被选中的码字梯度恒等于 0**。于是"某码字这一步没有商品"是**吸收态**——一旦进入就再也回不来。50 个 epoch 内 256 个一级码字死掉 193 个。

下限 `floor ≥ 1` 直接让这个状态不可达:任何一步的分配都不可能留下空码字。可行性条件是 `items ≥ codewords × floor`,不满足时**抛错**,不返回部分满足的结果。

## 用法

```bash
# 装进一棵候选树(自动识别量化器家族),然后立刻验证
python -m anticollapse <候选树路径>

# 只验证、不修改
python -m anticollapse <候选树路径> --check-only

# 保留 Sinkhorn,只给 argmin 路径加下限
python -m anticollapse <候选树路径> --no-force
```

成功时输出形如 `{"used": 8, "codewords": 8, "ok": true}` —— 验证方式是**在全新解释器里**把一棵 8 码字的量化器喂给"只有 3 个码字便宜、其余 5 个不可达"的代价矩阵,要求 8 个码字全部被使用。这正是本仓库里让所有臂坍缩的那种代价矩阵。

## 它改了哪里

| 家族 | 文件 | 改动 |
|---|---|---|
| `mixed` | `model/mixed_rq.py` | `MixedVQLevel._indices` 与 `encode_tokens` 的 `argmin` → `constrained_assignment(cost, ASSIGNMENT_FLOOR)`;可选关闭 Sinkhorn |
| `euclid` | `model/layers.py` | `VQLayer._indices` 与导出处的 `argmin` → 同上;可选关闭 Sinkhorn |

装进树里的是一个 60 行、只依赖 torch 的 `assignment_floor.py`(由本包的 `floor.py` 复制而来),加两个常量:

```python
ASSIGNMENT_FLOOR = 1          # 每个码字至少 1 个商品
FORCE_CONSTRAINED_ALL = True  # True: 关闭 Sinkhorn,训练与导出共用一条规则
```

`FORCE_CONSTRAINED_ALL = True` 很关键:只给训练侧加下限、导出仍走原来的规则,会得到"训练时 256 个码字、导出时仍在空码"的分裂(第 8 轮实测 H1 的 L3 训练/导出一致率只有 0.48)。

## 它现在是强制的

- **候选创建时自动安装**:`tools/research/rounds/baseline.py::copy_baseline` 在"副本与基线一致"的证明**之后**安装这一层,并把 `assignment_floor` 与安装后的源码摘要写进 provenance。经此创建的每个候选树天生带保护。
- **启动时兜底安装**:`stage2/launcher.py::launch`(§68 的唯一启动路径)在 spawn 之前再装一次,并把安装结果打到启动日志(`[stage2-launch] assignment_floor …`),**不写任何文件**(owner 2026-11-22)。即使树是旧的、或绕过 `copy_baseline` 生成的,也进不了训练。
- **调用点自检(不是信任)**:`verify_assignment` 在量化器的**返回处**检查真正返回的索引(`min(count) >= floor`),`verify_codebook` 在导出时拒绝两行完全相同的码本。机制若把分配改到别处(例如重新打开 Sinkhorn 分支),会在这一步被抓到,而不是静默失去保护。
- **平衡分配不关**:基线用 Sinkhorn(`SK_EPSILON=0.003`)时,下限施加在它的输出上(`argmax(q)` 取反后做受限分配),`use_sk` 保持基线原样(§19h)。校准:第 8、9 轮的树本来就没动这一行;第 10 轮改过,已在报告里更正。
- **§88 保持原样**:`run_all` 只跑 `check_entry`;`check_assignment_floor` 是独立诊断,不是准入闸门。保护是结构性的(创建时安装 + 启动时兜底),不靠新闸门。
- 因此"任何新机制都不坍缩"不再依赖作者记得跑命令,而是框架的属性。

## 一条命令检查一棵树

```bash
<python3.9> tools/research/stage2/anticollapse/selfcheck.py <tree> [--sid <item_sids.json>]
```

退出码 0 = 安装、三通路探测、产物审计全过;1 = 第一处失败的原因。完整的保证与边界见 `GUARANTEE.md`。

## 幂等与失败关闭

- 已经装过的树再调用一次是 no-op(`changed: false`)。
- 树里存在量化器文件但锚点对不上 → 抛 `UnknownTree`,**绝不静默跳过**。
- 没有任何已知量化器的目录 → 抛 `UnknownTree`。
- 目标是冻结基线(`stage2_RQ-VAE/baseline/...`)→ 抛 `ReadOnlyBaseline`,文件保持逐字节不变(§19b)。

## 边界

1. 它保证**占用**,不保证**质量**。编码器若把所有商品映射成同一向量,下限仍然可以被满足(把商品硬塞进去),此时必须由 `tools/research/qda/round7` 的认证闸门(有效秩、行为保真)拒绝。
2. 它不提高最终 SID 的行为保真。第 8 轮实测:H1/H2 的 L3 行为 AUC 分别比冻结基线低 0.39/0.54 pp,尽管唯一性最好(collision 0.083)。
3. 它只覆盖本仓库出现过的两种量化器实现;新写的量化器需要加一个 `Family` 条目。

## 测试

```bash
<python3.9> -m pytest tools/research/stage2/anticollapse/test_anticollapse.py -q   # 16 项
```

其中对**真实树**做端到端验证的有:复制冻结基线(`VQLayer` / 欧氏)与混合量化器、打补丁、在新解释器里跑坍缩代价矩阵要求 8/8;把码本改成两行相同后,打补丁的层在 eval 下抛 `CodebookCollapse`;`launch` 在 spawn 前确实安装了这一层;冻结基线被拒绝且 SHA 不变。
