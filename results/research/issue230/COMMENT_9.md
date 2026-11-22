## C22:冲突率表(按结构、按层)——结论是"单一前缀几何不可满足"只发生在跨结构,且规模很小

issue §3 要求:先分析训练 graphlet 的相互冲突(同一对商品被拉近/推远)以及硬前缀传递性导致的
**不可能约束**,再选机制。下面是测量。

### 口径

- 输入:冻结 supervision bank(实际进入 loss 的实例)与整个抽样 bank(20,000/结构),`banks.json` 记录
  sha256;两份表都产出(`conflicts.json`、`conflicts_full.json`)。
- `near`/`far` 位置来自 issue §2 的每条 Δ 公式,角色元组统一为 `(A,B,C)`,`A` 在位置 0:
  chain near=(AB,BC) far=(AC);fork near=(BC) far=(AB,AC);merge near=(BC) far=(AB,AC)。
- 无向商品对 `(min,max)`;"同一对被同时要求 near 和 far"即 issue 所说的不可能约束。

### 结构内(同一结构的两条实例把同一对拉向相反方向)

| 结构 | 监督 bank 冲突对数 | 全部抽样 bank 冲突对数 | near 对 | far 对 |
|---|---:|---:|---:|---:|
| chain | **0** | **0** | 19,772 | 19,526 |
| fork | **0** | **0** | 19,980 | 5,428 |
| merge | **0** | **0** | 19,992 | 6,430 |

这是采样器的设计后果,不是巧合:每种结构都只使用**严格诱导**的三节点子图(C1),一条边的两个端点
不可能同时是"相邻对"和"端点对"。全 20,000 条下也是 0,说明抽样没有把这个约束放松掉。

### 结构间(同一对是一条结构的 near 证据、另一条结构的 far 证据)

这是唯一真实存在的冲突,量很小:

| 冲突方向 | 对数(supervision) | near 侧实例数 | far 侧实例数 |
|---|---:|---:|---:|
| near(chain) ∩ far(fork) | 116 | 380 | 633 |
| near(chain) ∩ far(merge) | 103 | 163 | 332 |
| near(fork) ∩ far(chain) | 3 | 3 | 3 |
| near(merge) ∩ far(chain) | 2 | 2 | 2 |
| near(fork) ∩ far(merge) | 0 | 0 | 0 |
| near(merge) ∩ far(fork) | 0 | 0 | 0 |

fork 与 merge 之间**零冲突**:它们的 near/far 位置相同(只差边方向),方向本就不进入三码前缀
(issue §2 明确说方向不可从前缀恢复)。

### 对机制选择的直接含义

1. **结构内可满足**:每个结构各自有一条一致的目标,不需要"放弃某个结构"。
2. **跨结构冲突只占 chain 监督实例的 380/7651 = 5.0%、fork 的 633/6854 = 9.2%**,且在 fork↔merge 上为零。
   这意味着一个共享的前缀几何在数学上**是**可同时满足三个结构的,冲突质量不足以证明"不可满足"。
3. 因此机制不能靠"给结构让路"取胜,必须让**冲突的那一部分实例也能被满足** —— 即同一对商品在同一结构内
   用不同的共享关系区分开。三码前缀是等价关系,这要求把"角色"而不是"边方向"编码进前缀。

### 补充测量(C22 的另一半:硬前缀传递性的不可能约束)

三码共享前缀是等价关系:若 S(A,B)=3 且 S(B,C)=3,则 S(A,C)=3。因此**任何**要求
"S(A,B) 高、S(B,C) 高、S(A,C) 低"的实例都不可能满足。上一轮的 `pivot_witness.json` 已量化这一形态:
全网 **198,485** 对 `(A,C)` 同时满足 `A→B`、`B→C`、`C→A`(闭合三角形),它们在 chain 的 near 位置上
必然与 far 位置矛盾 —— 这正是"链式相邻对均值"判据(issue §2,C19 明确不要求两条严格)存在的原因。

证据:`results/research/issue230/conflicts.json`、`results/research/issue230/conflicts_full.json`、
`results/research/issue230/pivot_witness.json`、`results/research/issue230/banks/banks.json`。
