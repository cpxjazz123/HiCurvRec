# 对抗探测:到底哪些坍缩被挡住,哪些逃掉了

"任何机制都不会导致码本坍缩"这句断言,只有在能列出**敌手**并说明每个敌手的下场时才成立。本文件跑五个故意坍缩的量化器,全部经过**已安装保护**的真实树(冻结基线的副本,全新解释器)。

| 敌手 | 做了什么 | 树内判定 | 产物判定 | 谁接住了它 |
|---|---|---|---|---|
| `classic_collapse` | 用 round 1 的坍缩代价矩阵跑普通 `argmin` | **REPAIRED** 8 → 256 | — | 分配下限 |
| `duplicate_codebook` | 码本全部用上,但两行完全相同 | **REFUSED** | — | 导出时 `verify_codebook` |
| `bypass_tree` | 整棵树里把 `forward` 的分配换成自己的 `argmin` | **REFUSED** | — | 准入探测走 `forward`(不是只走 `_indices`),调用点自检当场抛 `CodebookCollapse` |
| `encoder_collapse` | 所有码字到所有商品等距,分配全靠平手 | **REFUSED** | — | 调用点 `verify_margins`:平手判定的商品占比 > 50% 即拒 |
| `frozen_codebook` | 码本不动、编码器训练(对照) | SURVIVED | — | 正确,这不是坍缩 |

## 结论

**码本层面的坍缩被构造性挡住:** 空码字不可能出现(分配下限),完全相同的码字在导出处被拒。

**五个敌手里四个被拒,唯一"存活"的是对照。**

两轮之前的两个逃逸现在都闭合:

1. **绕过分配**:准入探测走 `forward` 而非只走 `_indices`;替换 `forward` 里的分配会被调用点自检当场抛 `CodebookCollapse`。删掉校验行则 `apply_floor` 以 "the assignment check was removed" 拒绝重装。
2. **表示坍缩**:判据从"看起来用了 256 个码"改为 `verify_margins` —— 若超过半数商品的最近/次近代价差落在浮点分辨率内(即分配由平手而非数据决定),直接拒绝。**注意它是按平手判定的,不是按输入是否相同**:输入恒定但码字可分时代价并不平手,那不算坍缩(实测会通过)。

剩下的真实边界:平手判定之外**接近**坍缩(间隔极小但非零)不会被拒,那是第 7 轮认证闸门的职责。

## 复现

```bash
<genrec_env_v2>/python3.9 results/research/qda/round11/adversarial_probe.py
```

输出三列:树内判定、产物判定、接住它的层。`frozen_codebook` 是对照,用来证明该保护**不会**误伤"码本不动的机制"。
