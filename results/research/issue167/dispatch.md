# #167 派发记录

- 全门通过（见 `gates.md`）后派发：`stage2_start(source_dir="TIGER_af3_curvature_rq")`，worker 2322870。
- 冒烟（真实入口一 epoch）：`raw_unique=24587/24587`、**collision=0.000000** —— 条件配额的构造在 1 epoch 就已成立，不依赖训练收敛。
- 正式训练：seed 42、3000 epoch、全局 batch 4096、四卡 50202；产物 `results/stage2_RQ-VAE/TIGER_af3_curvature_rq/`。
- **速度记录（必须留痕）**：稳态 **5.5 s/epoch**（epoc 3–5 实测 5.78 / 5.55 / 5.54），约 **4.6 小时** 跑完 3000 epoch；基线是 0.27 s/epoch。成本来自每步的三层 Sinkhorn 计划、逐商品曲率距离（`distance_rows` 展开成 B×K 行）与配额指派。Issue 未设训练时间门，但这条数据对后续"是否值得"的判断是必需的。
