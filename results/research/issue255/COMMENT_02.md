C7/C8 基础设施完成：复用 `assets/manifest.json` 已登记资产，未重算、未复制、未覆盖任何既有文件。

复用（按不可变哈希）：具名图 `assets/train/visible_forward.npz` sha256 `ac7155d1424a6a346da9b8ed6e9413b81d43223a377308af0c8ba35b6b773029`、`assets/train/item_ids.npy` `262f7ce8…`、完整 1/2-hop `assets/v8_runtime_layout/`、全量曲率 `assets/shards/`+`item_summary.npz`+`node_moments.npz`、v9 ego 算子 `assets/v9_ego_operators/`（ego1/ego2 in/out CSR）、静态特征 `assets/v5_gnn_geometry/node_features.npz`、采样调度与 Triton 计算 `tools/issue253/*`。

新建（确实缺失才建）：`assets/v10_direction_ego/{in_only,out_only}/`，每个变体九个数组（nodes/offsets/node_counts/in_ptr/in_col/out_ptr/out_col/in_edge_counts/out_edge_counts），覆盖 24,587 商品、285,719 个 token 节点、695,377 条诱导有向边（每方向）。由 `tools/issue255/build_direction_ego.py` 生成，已登记进 `assets/manifest.json`（use=training，含 shape/dtype/bytes/sha256/来源/输入哈希）并在 `assets/README.md` 追加说明。独立逐商品重建审计 `assets/v10_direction_ego/audit.json`：两种变体各 24,587 项全部重算，mismatch=0，verdict=PASS。入集合与出集合重叠 11,800 项、互惠邻居 11,800 项、孤立商品 218/292 项——证明两类 token 人口并非互斥，且空邻居只保留 root、绝不伪造邻居。`tools/issue255/tests/test_direction_ego.py` 5 项测试（手算小图 + 真实资产重建 + manifest 哈希）通过。

模型侧：四类 token（完整1-hop、完整2-hop、中心+全部入邻居、中心+全部出邻居）经共享有向 GNN/readout，固定类型编码，专家注意力前不合并；逐层三专家门控输出 [N,3层,3专家]，每层专家维归一、token 维注意力归一；累计前缀 q_i^(l)=Σ_{j≤l}e_{i,j} 与实际 SID 前 l 码字逐位一致（已由测试断言）。CPU 前向冒烟：tokens [3,4,32]、weights [3,3,3]（每层专家和=1）、prefixes [3,3,32]、重建 [3,768]。

尚未完成（继续推进，非收尾）：候选全量测试套件、机器准入与四卡性能窗口、正式 3000 epoch Stage2、Stage3、机制指标与配对评估、报告/VERDICT/归档。
