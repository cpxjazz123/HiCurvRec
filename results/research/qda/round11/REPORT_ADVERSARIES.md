# 对抗探测:到底哪些坍缩被挡住,哪些逃掉了

"任何机制都不会导致码本坍缩"这句断言,只有在能列出**敌手**并说明每个敌手的下场时才成立。本文件跑五个故意坍缩的量化器,全部经过**已安装保护**的真实树(冻结基线的副本,全新解释器)。

| 敌手 | 做了什么 | 树内判定 | 产物判定 | 谁接住了它 |
|---|---|---|---|---|
| `classic_collapse` | 用 round 1 的坍缩代价矩阵跑普通 `argmin` | **REPAIRED** 8 → 256 | — | 分配下限 |
| `duplicate_codebook` | 码本全部用上,但两行完全相同 | **REFUSED** | — | 导出时 `verify_codebook` |
| `bypass_tree` | 整棵树里把 `forward` 的分配换成自己的 `argmin` | **REFUSED** | — | 准入探测走 `forward`(不是只走 `_indices`),调用点自检当场抛 `CodebookCollapse` |
| `encoder_collapse` | 所有商品向量相同,256 码全靠平手打破 | SURVIVED(256 码) | PASS(256/256/256) | **码本没坍缩,坍缩的是编码器**;由第 7 轮认证闸门(近/次近代价间隔≈0)拒绝 |
| `frozen_codebook` | 码本不动、编码器训练(对照) | SURVIVED | — | 正确,这不是坍缩 |

## 结论

**码本层面的坍缩被构造性挡住:** 空码字不可能出现(分配下限),完全相同的码字在导出处被拒。

**剩下的唯一逃逸是表示坍缩,它不在"码本"这一层:**

- 所有商品向量相同 → 代价矩阵处处相等 → 256 个码全靠平手规则分配。码本**形状上完全健康**(256/256/256、无重复行),信息量为零。
- 这不是码本坍缩,是编码器坍缩。它由第 7 轮的认证闸门拒绝(最近/次近代价间隔趋近 0),那是事后诊断,不属于本保护。

**上一版里"绕过分配"那个逃逸已经闭合**,闭合方式是让准入探测走 `forward` 而不是只走 `_indices`:机制若替换 `forward` 里的分配,调用点的 `verify_assignment` 会当场抛 `CodebookCollapse`,准入判 `FAIL`。若连校验行一起删掉,`apply_floor` 会以 "the assignment check was removed" 拒绝重新安装 —— 删掉保护与破坏保护在这套规则下等价。

## 复现

```bash
<genrec_env_v2>/python3.9 results/research/qda/round11/adversarial_probe.py
```

输出三列:树内判定、产物判定、接住它的层。`frozen_codebook` 是对照,用来证明该保护**不会**误伤"码本不动的机制"。
