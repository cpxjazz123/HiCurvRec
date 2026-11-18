# 300 epoch 实测:平手判据会不会在训练后期误杀

`verify_margins` 在"超过一半商品的归属由平手决定"时拒绝该步。这个 50% 的判据在初始化时显然安全,**训练后期码字互相靠近时是否仍安全,此前没有测过**。本实验补上。

## 做法

冻结基线 + 保护层,300 epoch(诊断预算,非正式预算),所有输出重定向到 `round14/run/`,不触碰任何已发表结果。保护层全程激活 —— 一旦平手比例越过 50%,该步即死。

## 结果:跑完了,而且离阈值很远

```
epoch=300 loss=0.00026949 recon=0.00009706 collision=0.104567
exported raw_shape=(24587, 3) unique=24587
```

最终 checkpoint 上的实测(全量训练集,逐级):

| 级 | 平手商品占比 | 中位间隔 | 最小间隔 |
|---|---|---|---|
| L1 | **0.0081 %** | 1.5e-3 | 1e-6 |
| L2 | **0.053 %** | 1.0e-3 | 0 |
| L3 | **0.15 %** | 4.8e-4 | 0 |

判据是 50 %,实测最高 0.15% —— **约 330 倍余量**。也就是说这条检查是"坍缩探测器",不是"训练约束":正常收敛离它很远,只有真正的退化才会碰到。

## 一次事故与修复(必须记录)

第一次跑这个诊断时,构建脚本只替换了 `ARM_NAME = "TIGER_RQ-VAE"` 的**旧格式**字符串,替换没生效,训练把导出写进了**已发表基线的结果目录**,覆盖了 `results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json`。

- 已从 HEAD 逐字节恢复(`git diff` 反向应用),`results/stage2_RQ-VAE/` 现无改动;
- 构建脚本改为**显式重定基所有输出路径**(`RESULT_DIR`/`OUTPUT_SID`/`OUTPUT_JSON`/`CHECKPOINT`/`LOG_DIR`/`METRICS_PATH`/`EXPORT_MANIFEST`),不再依赖 `ARM_NAME` 是否替换成功;
- 重跑后 `results/stage2_RQ-VAE/` 保持干净。

这与之前 `EXPORT_MANIFEST` 被 `check_entry` 覆盖是**同一类错误**:trainer 的路径常量在 import 时派生,事后改 `RESULT_DIR` 不会带走它们。凡运行诊断,必须逐个重定基并在跑完后核对结果目录无漂移。

## 复现

```bash
<genrec_env_v2>/python3.9 results/research/qda/round14/build_margin_run.py
cd results/research/qda/round14/trees/MARGIN_300 && <genrec_env_v2>/python3.9 -u train_rqvae.py
<genrec_env_v2>/python3.9 results/research/qda/round14/inspect_margins.py
```

日志 `round14/runs_stdout.log`;checkpoint 与导出在 `round14/run/`(不入库)。
