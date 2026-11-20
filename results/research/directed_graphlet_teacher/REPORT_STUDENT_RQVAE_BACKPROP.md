# Student 结构 Loss → 真实 RQ-VAE:训练连接、梯度与短程 smoke(Issue #223)

日期:2026-11-20
范围:只做训练连接与证据。#222 的 Teacher、硬标签、Student 头原样复用;不重训 Teacher、
不重建图/池、不新增几何/AF3/DGDV;未跑完整 3000 epoch,也未进入 Stage3。

## 0. 结论

1. **通路成立**:`L_structure` 经真实三层量化器反向,编码器与**三层码本原始参数**
   (`vq_layers.<i>.embed.weight`,不是 `get_code_embs()` 的中间张量)都收到非零梯度,
   一次真实 `optimizer.step()` 同时改变它们;`λ=0` 与 `λ>0` 的更新方向可测地不同。
   Teacher 梯度为 `None`、参数逐位不变;Student 输入 bundle 只有 `q`。
2. **代价极其不对称**:同一配置下 `L_structure` 的编码器梯度只有主损失的 ~1.9%
   (0.00147 vs 0.0762),码本侧 ~2%。所以"能接上"不等于"能训练"。
3. **smoke 是否定结果**:按真实 50-epoch warmup 形状跑 200 步,结构与无结构两臂的
   SID 碰撞几乎一致(0.378 vs 0.326),`L_structure` 只从 32.83 降到 32.22;在全学习率下
   跑 40 步,`L_structure` 确实降到 19.36,但三级 SID 坍缩(碰撞 0.916 vs 对照 0.456,
   L3 唯一码 2,067 vs 13,377)。**在码本安全的权重下它学不动,在它学得动的权重下码本坍缩。**
4. **框架缺口**:该臂的 `health_verdict` 只看 L1 最大桶与码字 spacing,碰撞 0.916 时仍判
   `pass`;§89 的 0.10 碰撞门槛是 Stage3 入场门,不是训练中门。`health_stop.json` 因此
   由"处理臂 vs 对照臂"的差值规则单独写出,不依赖该臂自带的门。

## 1. 接了什么

新增 `trainable_student.py`(薄集成层,不改动 Stage2 入口的既有逻辑):

```text
items(768) → model.encoder → z → model.rq(真实三层量化器)
          → hard tokens(真实分配) → q_st(hard 前向、soft onehot 反向)
          → StudentQuantisedHead → L_structure → backward
          → encoder 参数 + 三层 vq_layers.<i>.embed.weight
```

- 用 `q_st` 而不是量化器 `forward` 的输出:后者是 `x + (x_q - x).detach()`,
  只对编码器做 straight-through,**码本拿不到任何梯度**。`q_st` 的前向值等于所选码字之和
  (测试逐值对照 `hard_cumulative` 与 `get_indices`,32 项 > 16 码的可行域下)。
- 监督可用 #222 的冻结 Teacher(`torch.no_grad()` 的 KD)或真实硬标签 CE;两者走同一条
  Student→q→量化器→编码器通路。Teacher 输入的三层消息传递只出现在监督分支。
- `safe_aux_weight` 把预登记的 10% 份额做成**每个参数组的上界**:取
  `min(0.1·‖g_main,encoder‖/‖g_aux,encoder‖, min_level 0.1·‖g_main,ℓ‖/‖g_aux,ℓ‖)`。
  臂里原有的 `calibrate_weight` 只管编码器一侧,码本份额由它自由浮动。

## 2. 梯度与更新证据(`backprop_evidence.json`)

| 量 | 数值 |
|---|---|
| `L_structure` 单独反传 → 编码器梯度范数 | **0.00147** |
| 同上 → 三层码本梯度范数 | **0.00129 / 0.00123 / 0.00092** |
| 同上 → Student 头梯度范数 | 0.136 |
| 同上 → Teacher 梯度范数 | **0.0**(参数 SHA 前后逐位相同) |
| 主损失参考:编码器 / 三层码本 | 0.0762 / 0.0552 / 0.0668 / 0.0715 |
| 一次 AdamW(λ=0)后编码器 / 码本是否改变 | 是 / 是 |
| 一次 AdamW(λ=0.5)后编码器 / 码本是否改变 | 是 / 是 |
| λ=0 与 λ>0 的更新差(参数最大差) | 编码器 **0.00200**,码本 **0.00200** |

失败注入(全部在 `tests/test_trainable_student.py`,共 10 项):

| 注入 | 结果 |
|---|---|
| `detach_quantised=True`(量化值被 detach) | `LearningPathError`:编码器无梯度 |
| 把三层 `embed.weight` 设 `requires_grad_(False)` | `LearningPathError`:码本无梯度 |
| optimizer 不含量化参数 | `LearningPathError`:列出缺失组 |
| Teacher 参与反向 | `LearningPathError`:Teacher 收到梯度 |
| bundle 携带邻接/标签 | `LearningPathError`:禁止键 |

## 3. 短程 smoke(`smoke_trainable_student.json`,`health_stop.json`)

真实宽度 768→512→256→128→32、真三层码本(256×3)、真 `q_st`、真 `health_verdict`、
真 `codeword_spacing`、真 k-means `init_codebook`(与入口一致)。batch 512,seed 42。

| 运行 | 步数 | λ 权重 | L_structure | 三级碰撞 | L1 最大桶 | L3 唯一码 |
|---|---:|---:|---:|---:|---:|---:|
| warmup 对照 | 200 | 0 | 32.83 → 32.83 | **0.326** | 0.096 | 16,574 |
| warmup 结构 | 200 | 5.6e-5 | 32.83 → 32.22 | **0.378** | 0.122 | 15,294 |
| 全学习率 对照 | 40 | 0 | 32.83 → 32.82 | **0.456** | 0.087 | 13,377 |
| 全学习率 结构 | 40 | 7.6e-5 | 32.83 → **19.36** | **0.916** | 0.212 | **2,067** |
| 臂原校准(仅编码器份额) | 20 | 2.4e-4 | 32.83 → 31.85 | 0.861 | 0.380 | 3,406 |

三条必须同时读的事实:

1. **基线本身在漂**:k-means 初始化时碰撞是 0.031,对照臂在 100 步处也升到 0.741 再回落到
   0.326。batch 512 的前 200 步本来就不稳,所以不能把绝对碰撞全归给结构损失。
2. **结构臂恒差于对照**:warmup 端点 +0.052,全学习率端点 +0.460(L3 唯一码少 85%)。
3. **臂自带门没有拦住**:碰撞 0.916 时 `health_verdict` 仍为 `pass`——它不看三级碰撞。
   `health_stop.json` 记录的是"处理臂相对对照臂"的退化(全学习率:碰撞 +0.46、L1 最大桶
   0.212),这是本 Issue 认为应当停止的信号。

## 4. 判定与含义

- **工程交付成立**:结构损失确实经真实量化路径更新编码器与三层码本,`λ=0` 对照、Teacher
  冻结、detach/假量化/漏 optimizer 的注入测试齐备;smoke 有原始日志、健康轨迹与状态 SHA。
- **有效预算为零(在本配方下)**:按安全的 10% 份额上线,结构损失在 200 步里只降 1.8%,
  且 SID 与对照同带;把它放大到学得动(全学习率下自然权重 7.6e-5 已足够让它降 41%),
  L3 唯一码掉到 2,067,已是坍缩。这与 #221 D2–D4 的 >92% 碰撞是同一失效模式。
- **因此不进入完整 Stage2/Stage3**。#223 的 Done 定义只要求"结构 Loss 确实通过真实量化路径
  更新 Encoder 和三层码本 + 测试与 smoke 证据齐备",这两条都满足;有效性(先解决码本容量与
  监督之间的竞争)应单独立题,且必须先加**训练期三级碰撞门**——现有臂的门在 0.92 碰撞时仍放行。

## 5. 复现

| 入口 | 作用 |
|---|---|
| `trainable_student.py` | 训练连接、梯度分组、`safe_aux_weight`、`prove_update` |
| `evidence_backprop.py` | §2 的梯度/更新表 → `backprop_evidence.json` |
| `smoke_trainable_student.py` | §3 的三组 smoke → `smoke_trainable_student.json`、`health_stop.json` |
| `tests/test_trainable_student.py` | 10 项:梯度分组、更新差分、λ=0、detach/冻结/漏 optimizer 注入、hard 前向一致、无结构输入 |
| `manifest.py` | `MANIFEST.json` 记录上述文件与 checkpoint 的 sha256 |

测试:`/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 -m pytest tests/ -q`(25 项全绿)。
