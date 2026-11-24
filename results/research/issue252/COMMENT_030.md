## C9 实测定位并消除训练步的同步与阻塞搬运（数学不变）

单卡真实训练（batch 1024，同一模型/数据，预热后 8 步）剖析：

| 项 | 优化前 | 优化后 |
|---|---:|---:|
| 单步墙钟 | 128.8 ms | 101.7 ms |
| 其中 CPU 发射/等待 | 85.2 ms | 83.9 ms |
| 每步设备→主机同步 | 211 次 | 见下 |

逐步定位（每步计数）：`item` 176、`nonzero` 93、`where` 71、`arange` 66、`bincount` 60、`argsort` 46、`repeat_interleave` 36、`float` 26、`bool` 18、`sparse_csr_tensor` 16、`sparse.mm` 16、`cpu/numpy` 28。

五处改动，全部先红后绿、数值不变：

1. **ego 批次拼装**：布局加主机侧普查（每件节点数/每向边数），`repeat_interleave` 用 `output_size`、`arange` 用主机整数，`unique` 与身份校验移到主机。暖调用同步 **91 → 2**（RED 158/91/165，GREEN 33/2/4）。新增回归 `tests/test_sync_free_batch.py`。
2. **地板约束分配**（框架与候选两份副本逐字节一致）：修复轮改用定长 scatter（`bincount`/索引表 → 固定尺寸），分配结果与原实现**逐位相同**（`tools/issue252/test_assignment_floor.py`，10 个矩阵对照），每轮 1 次同步。
3. **编译 Sinkhorn**：归一化的 min/max 被当标量回读设备（`buf23.item()`），中心化移出编译区，数学不变。
4. **几何代价的 scale**：原 `bool(scale>0)` 对设备张量取值 → 每层一次回读（8 次/步、31 ms 等待）；改为冻结的主机数值（`tangent_scale`/`target_scale` 校准后即冻结），数值逐位不变。
5. **主机→设备搬运**：可分页拷贝会等 GPU 队列排空 → 改 pinned 暂存（异步）。

同时修复并复核资产链：ego operator 元数据按新生产者刷新（审计 PASS：24587 egos、1,716,410 / 513,330,917 有向边不变），manifest 同步，具名图契约 `verify_named_graph` **PASS**。

剩余 GPU 成本（2-hop 尺度，每批 145 万节点、2027 万边/向）：拼装 CSR 11.0 ms、两层 GNN 16.6 ms、反向 14.4 ms；1-hop 仅 ~12 ms；量化 27 ms；辅助总体 20 ms。按当前 4 卡 6 步/轮，0.5 秒/轮需要单步 ≤75 ms，现为 101.7 ms。

下一步：用同一套 60 epoch 连续窗口复测（诚实记录，不挑窗口），并继续压 2-hop 装配与 Sinkhorn 发射。正式训练仍未启动。相对证据路径均在 `results/research/issue252/`。