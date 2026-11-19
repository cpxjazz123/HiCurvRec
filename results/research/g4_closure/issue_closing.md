结项：四臂实验、冻结数据、CGV 六项、报告与源码全部入库（commit `36c06114a9`，main 已推送）。

**判定**：按本 issue 预设的停止条件——"若 D 与 B 持平、或结构增益以 SID collision/真边保留退化为代价，诚实记录没有证明高阶联合目标必要"——本轮记录为**未证明四节点闭合监督必要**。四种监督都没有把闭合/分叉的区分写进表征（ridge AUC ≤ 0.586，基线 0.578）或 SID 前缀（C/D 的 L1 gap 0.0022/0.0024，B 的 0.0325）；C/D 唯一的可测效应是把碰撞率压到 0.0136/0.0100，同时真实边保留与三角形保真掉到基线的 1/9 与 1/7。未启动 Stage3，未改任何门槛，不追加 AF3/曲率找补。

产物：`results/research/g4_closure/REPORT_G4_CLOSURE.md`（含四臂主表、probe、结构保真、CGV 六项、四个问题的直接回答与局限）；
`coverage.json` / `evaluation.json` / `probe.json` / `secondary.json` / `cgv/cgv_log.json` / `MANIFEST.json`；
四臂 SID 与 checkpoint 在 `results/stage2_RQ-VAE/TIGER_g4_closure_{A,B,C,D}/`（哈希见 MANIFEST）。

已知局限（报告中逐条记录）：单种子单次运行；分类头权重未入 checkpoint，故头在 heldout 上的 AUROC 由"表征 ridge 探针"替代；跨进程重编码与存档 SID 有 0.46%/0.60%/4.47%/0.53% 的行不同，原因是 `assignment_floor` 的码字使用计数是进程内状态。
