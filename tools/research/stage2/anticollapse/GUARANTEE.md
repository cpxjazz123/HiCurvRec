# 通用的码本防坍缩保证

一句话:**分配这一步被替换成"每个码字至少 1 个商品"的最便宜分配(码本大于 batch 时改为轮转覆盖),这个声明在调用点、导出处、产物处、准入处和静态扫描里被检查,机制不需要做任何配合。**

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
K > items（码字比一步的商品还多）                    轮转覆盖:每步填最久未用的码字,
                                                      ⌈K/items⌉ 步内每码至少用一次
```

- 只改分配。损失、距离、曲率、码本初始化、编码器、超参一行不动。
- 平衡分配(Sinkhorn,`SK_EPSILON=0.003`)保持基线原样,下限施加在**它的输出**上(§19h)。
- 可行性条件:`items >= K * floor` 走受限最近访;不满足时走**轮转覆盖**:每步把商品配给使用次数最少的 `items` 个码字,并推进该层的用量计数。上界:经过 `⌈K/items⌉` 步,每个码字至少被用过一次(实测 8 商品 × 64 码字 = 8 步覆盖 64/64)。
- 计数的可复现性:它是批次序列的纯函数 —— 同样的 seed 与数据顺序可复现;从 checkpoint 续跑会从零重计,只影响轮转相位。因此 `costs` 的调用必须带 `counts_key`(候选树里由补丁写成 `id(self)`);**不传计数的调用仍抛 `InfeasibleFloor`**,因为没有历史就无法轮转,不假装有保证。
- 轮转区间里的配对是"覆盖优先"而非"代价最优"(最优配对是 m×m 指派问题),这是明确写下的取舍。

**不可达性(构造性,不是惩罚项)**:第 `t` 步分配后必有 `counts >= floor >= 1`,因此"某码字没有商品"这个状态在每一步都被排除;吸收态无法被进入,也就不需要逃离。

## 3. 检查点

| 位置 | 检查 | 失败时 |
|---|---|---|
| 调用点(`forward` 内) | `verify_assignment(index, K, floor)`:看**真正返回的索引**。可行区要求每码 ≥ floor;轮转区(`items < K*floor`)要求商品互不共码,并报 `mode: rotation` | `CodebookCollapse`,该步死 |
| 调用点(`forward` 内) | `verify_margins(cost)`:平手判定的商品占比 > 50% 即拒(编码器未分离)。**DDP 下判定是全局的**:本地计数 `all_reduce` 后按全局比例决定,所有 rank 同时拒或同时继续 | `CodebookCollapse`,该步死 |
| 调用点(每步) | `verify_separation(codes, cost)`:码字间距相对**自身初始值**不得掉一个数量级(实测健康运行保有 0.51–0.75,被拉向均值的码本掉到 0.000)。DDP 下取各 rank 的**最小**比值再裁决,单 rank 阶段只报告 | `CodebookCollapse`,连续 50 步后 | 
| 调用点与导出 | `verify_codebook(codes)`:拒绝两行完全相同(`unique(dim=0)`,内存 O(K·D),每步都跑);近邻只报告(混合度量下 L3 实测 1e-4,属正常) | `CodebookCollapse` |
| 产物 | `audit_sid`:每级至少 2 个码字;三级码字重复率记录 | `check_entry` FAIL |
| 安装/准入(静态) | `floor_constant`:模块里的 `ASSIGNMENT_FLOOR` 必须存在且 ≥ 1。把它改成 0 会让所有守卫"通过"而分配退回裸 `argmin` —— 这是用一种整数写法删除保护 | `UnknownTree`: the floor was disabled |
| 安装/准入(静态) | `unguarded_sites`:量化器模块、`model/model.py`、`train_rqvae.py` 里**不允许残留**任何选码写法(`argmin` / `argmax` / `topk` / `.min(dim=)` / `.max(dim=)`;补丁会把它们替换掉);确非分配的用法必须在 `ALLOWED_SITES` 里带理由声明 | `UnknownTree`:报出文件与行号 |
| 准入 | `verify_tree`:对树里**每一个**量化器类(`VQLayer`、`EMAVQLayer`)各跑 **argmin / 平衡 argmax / 完整 forward** 三条通路,都要求 K/K | 准入 FAIL |

完整性检查对**已打过补丁的树也执行**:树只装一次,却要训练很多次,装完之后把常量改成 0、或再加一处分配,都必须被拒。

静态扫描是"任何机制"这一侧的通用化:它不枚举通路,而是要求**补丁后不再有任何未受保护的分配点**。上一版 EMA 那个洞(同文件第二个 `forward`)正是这条扫描会当场报出的东西,行号精确到 `model/layers.py:216`。

`VQ_TYPE` 允许 `vq`/`ema`/`simvq` 三种;`EMAVQLayer` 是同一个文件里的**第二个 forward**,必须单独打补丁并单独探测 —— 否则一个把 `VQ_TYPE` 设成 `ema` 的候选会在准入显示 PASS 的同时完全失去保护(实测:`EMAVQLayer` 的 forward 覆盖 1/8,被探测拒绝)。

配套:创建候选时安装(`copy_baseline`)、唯一启动路径兜底安装(`launch`)、安装文件过期自动刷新、删掉校验行拒绝重装(`the assignment check was removed`)、冻结基线拒绝改写(`ReadOnlyBaseline`)。

## 4. 实测证据

一条命令跑完九种形态(`acceptance.py`,约 10 秒,任一行不符即非零退出):

| 形态 | 判定 |
|---|---|
| 坍缩代价矩阵 + 普通 argmin | REPAIRED 8 → 256 |
| 重复码字 | REFUSED duplicate rows |
| 平手分配 | REFUSED tie-decided |
| 间距跌到 1/1000 | REFUSED spacing fell |
| 整树替换 `forward` 的分配 | REFUSED not guarded |
| 下限常量被改成 0 | REFUSED floor was disabled |
| 残留选码写法 `min(dim=)` | REFUSED model/layers.py:346 |
| 软分配 + 配方 helper | **PROTECTED** |
| 健康对照 | PROTECTED 8/8/8 |

训练级证据:

| 运行 | 结果 |
|---|---|
| 欧氏族,冻结预算 3000 epoch | 零拒绝,L1 死码 0,碰撞 0.1565,0.2089 s/epoch |
| 混合族(Sinkhorn 保留下限),300 epoch | 零拒绝,256/256/256,碰撞 0.0861 |
| 每层容量 [512,256,64],60 epoch | 零拒绝,512/256/64 全用 |
| 码本维度 16,60 epoch | 零拒绝,碰撞 0.0440 |
| 轮转区(`items < K`),10 epoch | 零拒绝,**该层 512 码字全用** |
| 注入坍缩(第 10 epoch 复制码字) | 下一步 `CodebookCollapse: 255 of 256` |

## 4b. 多卡

平手判定若只看本地 shard,可能出现"一个 rank 拒、其余 rank 在下一个集合通信里挂住"——拒绝会变成挂死。所以该判定在 `torch.distributed` 已初始化时做一次 `all_reduce`(计数与总数),按**全局**比例裁决;实测两 rank:仅一个 shard 越线时两 rank 都继续,两个 shard 都越线时两 rank 同时抛。

**只有全 rank 都进入的阶段才做集合通信**。冻结训练器在 eval 与导出时只跑 rank 0(`_skip_ddp_reduce = True`),那里若仍做 `all_reduce`,rank 0 会等待已经站在 barrier 上的其余 rank —— 拒绝又变成挂死。因此调用点把该标志传进来(`rank_local=...`),这种单 rank 阶段只**报告**不拒绝,判定留到下一个全 rank 步执行。导出锚点直接写死 `rank_local=True`(`encode_tokens` 是普通函数,没有 `self`)。

代价是每级每步一次 `all_reduce`(3 级即 3 次/步,仅在训练步)。**本机只有一张 GPU,四卡配方下的这次集合通信开销未被测量** —— 第一次四卡运行时应记录每 epoch 时间,与 round15 的单卡数字(0.2141 s/epoch)对照。
训练级证据:

## 5. 边界(必须写在任何"用了这个保护"的地方)

1. **它保证占用与区分性,不保证质量。** 退化编码器仍可满足下限(把商品硬塞进去),此时 SID 无信息量;判据是"码字有归属、互不相同、归属不靠平手、间距不塌",不是"表示有用"。
2. **接近坍缩由 `verify_separation` 判定,判据相对自身初始值**:码字最小间距 ÷ 量化器分辨率,跌破该层初始值的 0.1 倍并连续 50 步即拒绝。实测:健康 60 epoch 保有 0.506/0.691/0.754;3000 epoch 末态为 2.168/1.370/1.603(触发点约 0.030–0.044,**余量 ×42–×53**);被辅助损失拉向均值的码本为 0.000。局限:若某层**初始**码本就已退化,参考值接近 0,相对判据无从触发 —— 那种情况由精确检查(空码字、重复行)负责。
2. **接近坍缩由 `verify_separation` 判定,判据相对自身初始值**(因而是**度量无关**的:它比较码本间距与本次运行自己的分辨率,再与初始比值对照 —— 两种量纲的不匹配在比值里约掉,只有漂移会显现。实测角距代价下,码字向均值收缩到 0.01 倍时仍被拒):码字最小间距 ÷ 量化器分辨率,跌破该层初始值的 0.1 倍并连续 50 步即拒绝。实测:健康 60 epoch 保有 0.506/0.691/0.754;3000 epoch 末态为 2.168/1.370/1.603(触发点约 0.030–0.044,**余量 ×42–×53**);被辅助损失拉向均值的码本为 0.000。局限:若某层**初始**码本就已退化,参考值接近 0,相对判据无从触发 —— 那种情况由精确检查(空码字、重复行)负责。
3. **编码器坍缩按"平手"判定,不是按"输入相同"判定。** 超过半数商品的最近/次近代价差落在浮点分辨率内时拒绝:此时分配由平手而非数据决定。输入恒定但码字可分(代价不平手)不会被拒 —— 那确实不是坍缩。
4. **软分配可以受保护,但不是自动的**:按 5b 的配方(得分矩阵 + 经 `constrained_assignment` 出索引 + 可读码本)即可;自写 `argmax` 会被静态扫描以行号拒绝。
5. **`K > batch` 走轮转覆盖**,需要调用方传用量计数(`counts_key`);不传则抛 `InfeasibleFloor` —— 没有历史就无法轮转,不假装有保证。
6. ~~导出处检查依赖 `.eval()`~~ **已闭合**:重复码字检查是 `unique(dim=0)`,在每一步无条件执行。

## 5b. 怎么写一个能被保护的自定义量化器(含软分配)

守卫不认识"家族",只认识两样东西:**一张代价/得分矩阵**和**一个离散索引**。任何新量化器满足下面三条就能被保护,不满足就被拒绝(而不是被静默放行)。

1. **暴露代价或得分**:给出 `(items, codewords)` 的打分矩阵(软分配的 `q` 取负即可当作代价)。
2. **索引必须经由受限分配**:`index, _ = constrained_assignment(-q, ASSIGNMENT_FLOOR)`。这就是 Sinkhorn 分支的做法 —— 它本身就是软分配。
   - 若自己写 `q.argmax(...)`,静态扫描会以 `an assignment site is not guarded: <file>:<line>` 拒绝安装。
3. **码本要可读**:`get_code_embs()` 返回量化实际使用的那张表(原型+偏移也可以),否则 `verify_codebook` / `verify_separation` 无法检查。

实测(两向):

| 写法 | 结果 |
|---|---|
| 软分配 + `constrained_assignment(-q)` | 安装通过,下限成立,`verify_separation` 正常 |
| 软分配 + 自己的 `argmax` | 拒绝安装,并报出行号 |

## 5c. 两个默认关闭的开关(位置由 owner 决定)

这两条不改变保证,只改变判定**发生在哪里**;默认保持现状,打开即生效。

| 开关 | 位置 | 默认 | 打开后 |
|---|---|---|---|
| `checks.CHECK_ENTRY_ENFORCE_COLLISION` | 准入 | `False` | `check_entry` 用 §89 的 0.40 拒绝一 epoch 产物(通常过早,需要早期豁免规则) |
| `collapse_watch.py --stop <pid>` | 训练期 | 关 | 碰撞率越线即 terminate 该进程,省下剩余预算 |

实测:接近坍缩的产物默认 `PASS` 且报 `gate REFUSED`;打开开关后 `FAIL: three-code collision 0.930166 reaches the stage3 gate 0.4`。

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
<python3.9> tools/research/stage2/anticollapse/acceptance.py                       # 9 形态矩阵
<python3.9> -m pytest tools/research/stage2/anticollapse/test_anticollapse.py -q   # 39 项
<python3.9> -m pytest tools/research/stage2/tests/test_contract.py -q              # 10 项
```

覆盖:可行性契约、两族真实树的端到端、重复码字端到端、整树绕过被拒、删校验被拒、基线只读、平衡规则不被改动、产物审计(含后缀列)、安装刷新、`launch` 兜底。
