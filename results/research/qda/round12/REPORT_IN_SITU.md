# 现场证据:保护在冻结训练器里是活的

单元测试只能证明"把一张重复码本交给我,我会拒绝"。这份报告证明的是**同一件事发生在真实训练循环里**。

## 做法

`build_in_situ.py` 从冻结基线复制一棵树、装上下限,再注入一个"机制":第 10 个 epoch 结束时把码字 0 复制到其余全部码字 —— 这正是写坏的码本重归一化会做的事。

## 实测

```
[RecBole RQ-VAE] initializing codebooks with KMeans
[RQ-VAE] epoch=1 loss=0.00052134 recon=0.00046699 ...
...
[RQ-VAE] epoch=9 loss=0.00032454 recon=0.00025268 ...
[collapse] duplicated codeword 0 across every codeword
assignment_floor.CodebookCollapse: 255 of 256 codewords are duplicate rows
```

三点:

1. **前 9 个 epoch 正常训练**,没有任何误报 —— 下限、平手判定、重复码字检查在正常训练里都不出手。
2. **坍缩发生的下一 step 就被拒**,不是等到导出,也不是等到评估。
3. 报出的是**具体缺陷**(255/256 重复),不是一句笼统的失败。

检查顺序也调过一次:原先 `verify_margins` 先跑,全坍缩时它会报"100% 平手"—— 正确但不具体;现在 `verify_codebook` 在前,报"255/256 是重复行"。

## 复现

```bash
<genrec_env_v2>/python3.9 results/research/qda/round12/build_in_situ.py
cd results/research/qda/round12/trees/COLLAPSE_AT_10
<genrec_env_v2>/python3.9 -u train_rqvae.py      # 约 60 秒后以 CodebookCollapse 退出
```

日志留在 `results/research/qda/round12/runs/stdout.log`。该臂是诊断臂:结果目录与源码树在跑完后已清理,不入库。
