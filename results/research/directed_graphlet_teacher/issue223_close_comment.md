## 结项:结构 Loss 已接进真实 RQ-VAE 计算图;训练连接成立,有效预算为零(按 §Done 交付)

不重开 #221/#222,不跑完整 Stage2/Stage3。全部证据在 `results/research/directed_graphlet_teacher/`
(`MANIFEST.json` 逐文件 sha256;报告 `REPORT_STUDENT_RQVAE_BACKPROP.md`)。

### 1. Done 定义的两条都满足

**通路**:`L_structure` → `StudentQuantisedHead` → `q_st`(hard 前向 / soft onehot 反向) →
真实三层量化器 → 编码器与**三层码本原始参数**(`vq_layers.<i>.embed.weight`)。
量化器 `forward` 的 straight-through 只到编码器(`x + (x_q - x).detach()`),码本走不通,
所以码本侧必须用 `q_st`;其前向值已逐值对照 `hard_cumulative` 与 `get_indices`。

| 量 | 数值 |
|---|---|
| `L_structure` 单独反传 → 编码器梯度 | **0.00147** |
| 同上 → 三层码本梯度 | **0.00129 / 0.00123 / 0.00092** |
| 同上 → Student 头 / Teacher | 0.136 / **0.0**(Teacher 参数 SHA 逐位不变) |
| 主损失参考:编码器 / 三层码本 | 0.0762 / 0.0552 / 0.0668 / 0.0715 |
| 一次 AdamW 后编码器 / 码本是否改变 | 是 / 是(λ=0 与 λ>0 都是) |
| λ=0 与 λ>0 的更新差 | 编码器 0.00200、码本 0.00200 |

**测试**(`tests/test_trainable_student.py`,10 项,全套 25 项全绿):梯度分组、真实 optimizer
更新、λ=0 对照、hard 前向一致、无结构输入,以及四种伪造方式的失败注入——`detach` 量化值、
冻结码本、optimizer 漏掉量化参数、Teacher 收到梯度,全部按预期抛 `LearningPathError`。

### 2. 短程 smoke(真实宽度 768→…→32、真 256×3 码本、真 k-means `init_codebook`、真健康门)

| 运行 | 步数 | 权重 | L_structure | 三级碰撞 | L1 最大桶 | L3 唯一码 |
|---|---:|---:|---:|---:|---:|---:|
| warmup 对照 | 200 | 0 | 32.83 → 32.83 | 0.326 | 0.096 | 16,574 |
| warmup 结构 | 200 | 5.6e-5 | 32.83 → 32.22 | 0.378 | 0.122 | 15,294 |
| 全学习率 对照 | 40 | 0 | 32.83 → 32.82 | 0.456 | 0.087 | 13,377 |
| 全学习率 结构 | 40 | 7.6e-5 | 32.83 → **19.36** | **0.916** | 0.212 | **2,067** |
| 臂原校准(仅编码器份额) | 20 | 2.4e-4 | 32.83 → 31.85 | 0.861 | 0.380 | 3,406 |

三条读法:

1. **基线本身在漂**:k-means 初始化时碰撞 0.031,对照臂在 100 步也升到 0.741 再回落 0.326;
   batch 512 的前 200 步本来不稳,不能把绝对碰撞全记在结构损失头上。
2. **结构臂恒差于同步骤对照**:warmup 端点 +0.052;全学习率端点 +0.460,L3 唯一码少 85%。
3. **臂自带门不在看三级碰撞**:碰撞 0.916 时 `health_verdict` 仍 `pass`(它只看 L1 桶与 spacing)。
   `health_stop.json` 记录的是"处理臂 vs 对照臂"的退化,不依赖该臂的门。

### 3. 判定

- **工程交付成立**:结构损失确实经真实量化路径更新编码器与三层码本,`λ=0` 对照、Teacher 冻结、
  四种伪造注入、hard 前向一致与导出一致性都有可执行证据。
- **在本配方下有效预算为零**:按安全的 10% 份额(全组上界)上线,200 步里结构损失只降 1.8%
  且 SID 与对照同带;放到学得动的权重,三级码本坍缩(0.916 碰撞 / 2,067 个 L3 唯一码)。
  这与 #221 D2–D4 的 >92% 碰撞是同一失效模式。
- **不启动完整 Stage2/Stage3**。后续若继续,必须先加**训练期三级碰撞门**(现有门会放行 0.92),
  再研究"码本容量与结构监督如何不互相吃掉",而不是继续加大 λ。

### 4. 交付清单

`trainable_student.py` / `evidence_backprop.py` / `smoke_trainable_student.py` /
`tests/test_trainable_student.py` / `backprop_evidence.json` / `smoke_trainable_student.json` /
`health_stop.json` / `REPORT_STUDENT_RQVAE_BACKPROP.md` / `MANIFEST.json`;
提交 `bd6facba`(已推送,main)。
