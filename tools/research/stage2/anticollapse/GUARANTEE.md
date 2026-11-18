# 通用的码本防坍缩保证

一句话:**分配这一步被替换成"每个码字至少 1 个商品"的最便宜分配,并且这个声明在调用点、导出处、产物处和准入处被检查四次;机制不需要做任何配合。**

## 1. 什么算"码本坍缩"

| 形态 | 定义 | 为什么致命 |
|---|---|---|
| 空码字 | 某个码字在某一步一个商品都没有 | 该码字的码本项拿到**恰好为零**的梯度(码本项对残差 detach),再也回不来 —— round 1 实测 50 epoch 内 193/256 |
| 重复码字 | 两个码字的向量完全相同 | 两个码字对模型不可区分,容量凭空损失 |
| 共码 | `K > batch` 时多个商品被迫共享一个码字 | 该步可用的码字少于商品数,覆盖无法展开 |
| 单码层 | 某一级在整个导出里只用一个码字 | 该级是常数,等价于没有这一级 |

## 2. 机制

```
分配 = argmin over { a : counts(a) >= floor }          （可行区,K <= items）
K > items（码字比一步的商品还多）                    （**拒绝运行**,不是降级）
```

- 只改分配。损失、距离、曲率、码本初始化、编码器、超参一行不动。
- 平衡分配(Sinkhorn,`SK_EPSILON=0.003`)保持基线原样,下限施加在**它的输出**上(§19h)。
- 可行性条件:`items >= K * floor`。不满足时抛 `InfeasibleFloor`,**拒绝运行**,报出两边的数字与补救方向(减码字或加 batch)。
- 为什么不是"降级成每商品独占一码":那个分配是代价矩阵的纯函数,不会轮转(实测 8192 码 / 4096 商品 / 代价不变,连跑 10 步只用到同样的 4096 个);要轮转就必须记录每个码字的历史,分配于是依赖调用顺序,可复现性丢失。**要么保证,要么不训练** —— 这条路径选择了后者。

**不可达性(构造性,不是惩罚项)**:第 `t` 步分配后必有 `counts >= floor >= 1`,因此"某码字没有商品"这个状态在每一步都被排除;吸收态无法被进入,也就不需要逃离。

## 3. 四处检查

| 位置 | 检查 | 失败时 |
|---|---|---|
| 调用点(`forward` 内) | `verify_assignment(index, K, floor)`:看**真正返回的索引**;`items < K*floor` 本身就是失败 | `CodebookCollapse`,该步死 |
| 调用点(`forward` 内) | `verify_margins(cost)`:平手判定的商品占比 > 50% 即拒(编码器未分离)。**DDP 下判定是全局的**:本地计数 `all_reduce` 后按全局比例决定,所有 rank 同时拒或同时继续 | `CodebookCollapse`,该步死 |
| 调用点与导出 | `verify_codebook(codes)`:拒绝两行完全相同(行字节哈希,每步都跑);近邻只报告(混合度量下 L3 实测 1e-4,属正常) | `CodebookCollapse` |
| 产物 | `audit_sid`:每级至少 2 个码字;三级码字重复率记录 | `check_entry` FAIL |
| 安装/准入(静态) | `floor_constant`:模块里的 `ASSIGNMENT_FLOOR` 必须存在且 ≥ 1。把它改成 0 会让所有守卫"通过"而分配退回裸 `argmin` —— 这是用一种整数写法删除保护 | `UnknownTree`: the floor was disabled |
| 安装/准入(静态) | `unguarded_sites`:量化器模块、`model/model.py`、`train_rqvae.py` 里**不允许残留**任何选码写法(`argmin` / `argmax` / `topk` / `.min(dim=)` / `.max(dim=)`;补丁会把它们替换掉);确非分配的用法必须在 `ALLOWED_SITES` 里带理由声明 | `UnknownTree`:报出文件与行号 |
| 准入 | `verify_tree`:对树里**每一个**量化器类(`VQLayer`、`EMAVQLayer`)各跑 **argmin / 平衡 argmax / 完整 forward** 三条通路,都要求 K/K | 准入 FAIL |

完整性检查对**已打过补丁的树也执行**:树只装一次,却要训练很多次,装完之后把常量改成 0、或再加一处分配,都必须被拒。

静态扫描是"任何机制"这一侧的通用化:它不枚举通路,而是要求**补丁后不再有任何未受保护的分配点**。上一版 EMA 那个洞(同文件第二个 `forward`)正是这条扫描会当场报出的东西,行号精确到 `model/layers.py:216`。

`VQ_TYPE` 允许 `vq`/`ema`/`simvq` 三种;`EMAVQLayer` 是同一个文件里的**第二个 forward**,必须单独打补丁并单独探测 —— 否则一个把 `VQ_TYPE` 设成 `ema` 的候选会在准入显示 PASS 的同时完全失去保护(实测:`EMAVQLayer` 的 forward 覆盖 1/8,被探测拒绝)。

配套:创建候选时安装(`copy_baseline`)、唯一启动路径兜底安装(`launch`)、安装文件过期自动刷新、删掉校验行拒绝重装(`the assignment check was removed`)、冻结基线拒绝改写(`ReadOnlyBaseline`)。

## 4. 实测证据

敌手(全部经过受保护的**真实树**,全新解释器):

| 敌手 | 判定 |
|---|---|
| 坍缩代价矩阵 + 普通 argmin | REPAIRED 8 → 256 |
| 整树替换 `forward` 的分配 | REFUSED(调用点自检) |
| 重复码字 | REFUSED(导出检查) |
| 码本不动(对照) | SURVIVED(不误伤) |
| 编码器坍缩(全商品同向量) | SURVIVED —— 见边界 |

真实运行:`copy_baseline` → 一轮真训练(平衡开着)→ `PASS`,产物 `256/256/256`;混合族与欧氏族在三条探测通路上都是 `8/8`。

## 4b. 多卡

平手判定若只看本地 shard,可能出现"一个 rank 拒、其余 rank 在下一个集合通信里挂住"——拒绝会变成挂死。所以该判定在 `torch.distributed` 已初始化时做一次 `all_reduce`(计数与总数),按**全局**比例裁决;实测两 rank:仅一个 shard 越线时两 rank 都继续,两个 shard 都越线时两 rank 同时抛。

代价是每级每步一次 `all_reduce`(3 级即 3 次/步)。**本机只有一张 GPU,四卡配方下的这次集合通信开销未被测量** —— 第一次四卡运行时应记录每 epoch 时间,与 round15 的单卡数字(0.2141 s/epoch)对照。

## 5. 边界(必须写在任何"用了这个保护"的地方)

1. **它保证占用,不保证质量。** 退化编码器仍可满足下限(把商品硬塞进去),此时 SID 无信息量。
2. **编码器坍缩按"平手"判定,不是按"输入相同"判定。** 调用点 `verify_margins` 在**超过半数商品的最近/次近代价差落在浮点分辨率内**时拒绝:此时分配由平手而非数据决定,码本形状再健康也没有信息量。输入恒定但码字可分(代价不平手)不会被拒,这是正确的 —— 那确实不是坍缩。**接近**坍缩(间隔极小但非零)不在内,由第 7 轮认证闸门负责。
3. **协议要求离散索引。** 纯软分配量化器会被准入以 `UnknownTree` **拒绝**(失败关闭),而不是静默失去保护。
4. ~~导出处检查依赖 `.eval()`~~ **已闭合**:重复码字改用行字节哈希(`O(K·D)`,约 0.05 ms),现在在每一步都无条件检查,不再依赖训练/评估模式。

## 6. 如何使用

```bash
# 新建候选:自动安装并记录 provenance
from rounds.baseline import copy_baseline
copy_baseline(destination)

# 任何既有树:一条命令给出 PROTECTED / NOT PROTECTED
python tools/research/stage2/anticollapse/selfcheck.py <tree> [--sid <item_sids.json>]
```

`selfcheck.py` 退出码 0 表示安装、三通路探测、产物审计全部通过;1 表示第一处失败的原因。

## 7. 测试

```bash
<python3.9> -m pytest tools/research/stage2/anticollapse/test_anticollapse.py -q   # 21 项
<python3.9> -m pytest tools/research/stage2/tests/test_contract.py -q              # 7 项
```

覆盖:可行性契约、两族真实树的端到端、重复码字端到端、整树绕过被拒、删校验被拒、基线只读、平衡规则不被改动、产物审计(含后缀列)、安装刷新、`launch` 兜底。
