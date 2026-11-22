## Round 1 — 评估合同冻结与首批证据

**issue 正文冻结**：`results/research/issue230/issue230.json`，sha256 `301c5a9a…bfc7b0a0`（3,048 字符，OPEN，updatedAt 2026-11-22T01:16:01Z）。

**验收清单**：共 49 条（C1–C49），每条补齐判据/阈值/数据/口径，见 `results/research/issue230/RUN_LOG.md`。

**是否需要训练**：需要。依据 §5「三种 Δ 都高于同协议原始基线」——Δ 只能由 3000-epoch 训练导出的硬 SID 计算；Stage3 不跑（§5「本 Issue 不要求启动 Stage3」）。

### 本轮完成（有证据）

| 项 | 文件 | 证据 |
|---|---|---|
| 统一角色定义（A=中心；chain A→B→C；六条有向位置全检查） | `analysis_sid_recall/graphlet_contract.py` | 13 个单测全绿；旧 tuple `(B,C,A)` 被断言必须不合格 |
| Δ 口径（链用相邻对均值，S() 只读前三码） | 同上 | 人手算例 `(3,1,2)→0`、`(2,3,1)→1.5` |
| 严格诱导采样 | 同上 `sample()` | 真实有向图上三结构各 20,000 条全部严格诱导，`induced_audit.json` |
| 无对重叠的三向切分 | `analysis_sid_recall/graphlet_banks.py` | 切分后 `supervision∩validation / ∩test / validation∩test` 全为 0；冲突实例丢弃并计数 |
| 切片内互斥质量 | `analysis_sid_recall/graphlet_conflicts.py` | 见下 |

**Bank 规模**（`results/research/issue230/banks/banks.json`）

| kind | supervision | validation | test | 丢弃 |
|---|---:|---:|---:|---:|
| chain | 7,651 | 1,285 | 1,222 | 9,842 |
| fork | 6,854 | 820 | 788 | 11,538 |
| merge | 6,322 | 957 | 978 | 11,743 |

**结构间互斥**（同一无序对：一个结构要它 near、另一个要它 far，`conflicts_full.json`）

| 方向 | 对数 |
|---|---:|
| near(chain) ∩ far(fork) | 325 |
| near(chain) ∩ far(merge) | 394 |
| near(fork) ∩ far(chain) | 17 |
| near(merge) ∩ far(chain) | 18 |

### 未完成（下一步）

C12（checkpoint 重编码核对）、C14（基线 Δ 重算）、C22 的层内互斥口径核对、C33–C49（机制与训练）。C3/C4 的"训练侧与三个分析脚本切到合同模块"仍在进行。
