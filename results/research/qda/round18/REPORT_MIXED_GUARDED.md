# 混合族 + 平衡分配:300 epoch 实测

欧氏族已经跑到完整预算;混合族 —— 也就是最后一级用 Sinkhorn 平衡的那一族 —— 自"守卫不再关闭平衡"之后还没跑过。这轮补上。

## 设置

F0 研究树、装守卫、**平衡保持基线原样**(下限施加在 `argmax(q)` 的输出上),300 epoch,输出全部重定基到本轮目录。

## 结果

```
epoch=300 loss=0.00015137 recon=0.00010157 raw_unique=22471/24587 collision=0.086062
产物审计: PASS  levels 256/256/256  collision 0.086062  gate allowed
稳态每 epoch 0.3941 s
```

- **300/300 epoch 完成,零拒绝** —— 平衡分配被施加下限后仍然稳定。
- 三级码碰撞率 **0.086**;产物审计通过,每级 256 个码字全在用。
- 与第 10 轮那次(同族、当时守卫**关掉**了平衡、另有有界几何)的 0.2901 相比数值好得多,但那不是受控对比:机制不同。这里只说明**保留下限 + 保留平衡**这条路径本身产出健康的标识符。

## 现在的覆盖矩阵

| 族 | 分配规则 | 规模 | 结果 |
|---|---|---|---|
| 欧氏(`VQLayer`/`EMAVQLayer`) | argmin + 下限 | **3000 epoch** | 完成,L1 死码 0,碰撞 0.1557 |
| 混合(`MixedVQLevel`) | **Sinkhorn 输出 + 下限** | 300 epoch | 完成,256/256/256,碰撞 0.0861 |
| 两族 | 注入式坍缩 | 10 / 60 epoch | 下一步 `CodebookCollapse` |
| 两族 | 接近坍缩 | 60 epoch | 放行,由 Stage3 门槛拒绝(0.930) |

## 复现

```bash
<genrec_env_v2>/python3.9 results/research/qda/round18/build_mixed_run.py
cd results/research/qda/round18/trees/MIXED_GUARDED_300 && <genrec_env_v2>/python3.9 -u train_rqvae.py
```

约 2.5 分钟。
