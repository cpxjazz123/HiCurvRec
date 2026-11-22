## C32/C33:全局隔离 bank 已重建,**隔离通过,但监督量掉到 1/7**

### 为什么必须重建

`results/research/issue230/banks/` 只做了各结构**内部**隔离。按 C33「三种结构合并后的全局无序对交集必须为零」实测:

| 检查 | #230 bank | C33 要求 |
|---|---:|---|
| chain ∩ fork(supervision) | 119 | 0 |
| chain ∩ merge(supervision) | 105 | 0 |
| fork ∩ merge(supervision) | 37 | 0 |
| chain supervision ∩ 任意 eval | 122 | 0 |
| fork supervision ∩ 任意 eval | 88 | 0 |
| merge supervision ∩ 任意 eval | 69 | 0 |

**FAIL**,所以重建。

### 做法

`analysis_sid_recall/graphlet_banks_global.py`:

1. 对三种结构 **union** 后的全部无序对做一次全局划分(60/20/20);
2. 一个实例只有在它的**三对同属一个划分**时才被保留,否则丢弃并计数;
3. 再显式 **claim**:一对被某结构在 supervision 里占用后,其它结构的该行被移除 ——
   因为单靠划分无法保证(同一对可以同时落在两个被保留的不同结构实例里)。

### 结果

| 结构 | supervision | validation | test | 丢弃 | claim 移除 |
|---|---:|---:|---:|---:|---:|
| chain | **1,138** | 192 | 151 | 15,373 | 3,146 |
| fork | **1,059** | 149 | 164 | 15,441 | 3,187 |
| merge | **1,189** | 162 | 159 | 15,244 | 3,246 |

**隔离审计全部为 0**:

| 检查 | 值 |
|---|---:|
| chain ∩ fork(supervision) | **0** |
| chain ∩ merge(supervision) | **0** |
| fork ∩ merge(supervision) | **0** |
| supervision ∩ eval(三结构) | **0 / 0 / 0** |
| validation ∩ test(三结构) | **0 / 0 / 0** |

### 必须一起读的代价

监督量从每个结构约 7,600 降到约 1,100,**只剩 15%**。测试集也只剩 151/164/159 个实例。

- 这直接压缩了 §5 判据的统计功效:测试集约 155 例,远小于 #230 的 1,222/788/978;
- C44 要求"不得由单次半集差值宣称噪声底" —— 在 n≈155 上做聚类 bootstrap,区间会很宽,
  规格上必须如实报告宽度,不能因为宽就换口径。

这是 C33 与统计功效之间的**真实张力**,不是可以两全的:隔离越严,评估样本越少。
本轮按 issue 原文执行 C33(隔离优先),并把功效损失写进报告。

证据:`results/research/issue231/banks/banks_global.json`(含 12 份 bank 的 sha256)、
`analysis_sid_recall/graphlet_banks_global.py`。
