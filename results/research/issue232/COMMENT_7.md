## §5 结论对账 — **§4 FAIL,阻塞点是框架护栏与结构梯度尺度不相容**

`VERDICT.md` 已写,**65 条逐条判定**。issue 保持 **OPEN**。

### 判定汇总

| 组 | PASS | FAIL | NOT_RUN / PARTIAL |
|---|---|---|---|
| A 固定规则 C1–C7 | 7 | 0 | 0 |
| B 真实码字 C8–C20 | 12 | 0 | C19 **NOT_RUN**(无有效 checkpoint) |
| C 分层设计 C21–C36 | 15 | **C35 FAIL**(重建 1.061×) | 0 |
| D 训练前验证 C37–C44 | 9 | 0 | 0 |
| E bank/隔离 C45–C52 | 8 | 0 | C47 **PARTIAL**、C50 **PARTIAL** |
| F 验收 C53–C65 | 7 | **C54/C55/C56/C58/C61 FAIL** | 0 |

### 为什么 §4 的六项 FAIL 不能当作"机制失败"

| 轮次 | 结果 |
|---|---|
| run1(3000 epoch) | 结构项 **18,000 步恒为 1.3**,与 TIGER 三码 0/24,587 相同 = 基线 + assignment floor |
| run2(T=0.0016,A 方向) | 16 epoch 被 `CodebookCollapse` 拦下 |
| run3(方案二 encoder-only) | 32 epoch 被拦下;码本实测一步未动,encoder 侧崩 |

### 方案二的实测:encoder 侧没有稳定步长

| 步长预算 | L1 separation | 结果 |
|---|---|---|
| 无预算 | 0.5211 → 0.1626 | 崩 |
| fraction=0.10 | — | `verify_margins`: **tie-decided 94.5%** |
| fraction=0.030 | 0.4632 → 0.0000 | 崩 |
| fraction=0.010 | 0.4632 → 0.0091 | 崩 |
| fraction=0.003 | 0.4632 → 0.0096 | 崩 |

**单步实测**:lr=1e-3 下 encoder 一步把 latent 挪 **0.110**,分配分辨率仅 **0.0728** —— 位移是分辨率的 **1.5 倍**。
`GRADIENT_RMS=1` 把梯度 RMS 拉到 1.0 后,AdamW 对 562,080 个参数各走一步,累积位移必然超过码字间距。
**降低 lr 无效**:`verify_separation` 是 `码字最小间距 / 分配半径` 的**比值**,商品被推近时分母同步缩小。

### 阻塞(§7)

继续需要改框架判据,超出本轮授权:

1. 重新标定 `assignment_floor` 的两道护栏(`verify_separation` / `verify_margins`)——它们属于框架资产;
2. 或把监督改成不触碰 encoder 参数的形式(如导出阶段受限重指派)——这是另一个实验设计。

### 可交付物

- `stage2_RQ-VAE/TIGER_export_matched_graphlet_rq/`(从冻结基线复制,机制层逐文件可解释)
- `losses/layered_graphlet.py`、`losses/layered_wiring.py`、**23 项单测全绿**
- 四卡准入 PASS ×2;`patch_tree.py` anchor 修复在副本上验证
- `results/research/issue232/{VERDICT.md, metric.json, hashes.json, bank_audit.json, measure.py, run1_backup/}`
