# Task #180 — Idea 2 graph-aware encoder (LightGCN Step 1)

> **任务目的**: RQ-VAE encoder 之前加一层 LightGCN-style 邻居传播, 让 item embedding 在进 quantizer 之前混入共现图结构, 这时 ORC 才有资格去指导曲率选择. Step 1 不引入 κ 叠加, 仅验证共现图结构对 Stage 1 编码是否有正向价值. Step 2 (κ 叠加) 等 Step 1 出信号再决定.

> **完成日期**: in progress
> **状态**: 🟡 Phase 3 待启动 (R10 主动推进: 不空闲等待 Phase 1 Stage 3)

---

## 1. 背景

**2026-07-25 用户前置指令**: "Idea 2 (Step 1): RQ-VAE encoder 之前加一层 LightGCN-style 邻居传播, 让 item embedding 在进 quantizer 之前混入共现图结构, 这时 ORC 才有资格去指导曲率选择. Step 2 后续再做曲率叠加."

**前置 task 结论**:
- Task #69 (5-graph weight diagnose): G1 interaction HR@20=0.578 ⭐ 最佳 vs G4 full_kg 0.396 (-32%); 验证 R1-A/B/C/D 成立
- Task #70 (Ollivier curvature): Toys 数据本质是双曲 (-0.65 to -0.84, 99%+ 边 κ<0); 但 κ3 模型正则化收敛到欧式
- Task #178 (Phase 1, in progress): 修正后 HG-Rec baseline

**假设 R1**: 共现图结构对 Stage 1 编码有正向价值, LightGCN 1 层传播能补足 item_emb 仅含 semantic 的不足
**假设 R2**: 若 R1 失败 (R@10 ≤ 新 baseline ±0.005), LightGCN 1 层不构成 semantic 加成, 不再做 Step 2 κ 叠加

## 2. 实验设计

**变量**: Stage 1 encoder 之前加 1 层 LightGCN 邻居传播 (基于 G2 共现图)
**保持不变**:
- 共现图 = Task #69 G2 (5-window 双方向), 用当前 Instruments `train.parquet` 重生成
- HRQ-VAE 主体 = Phase 0 修复版 (utils.py 4 处 bug + train_hrqvae.py 论文原配)
- T5-small 5.5M (跟 #178 / Task #84 baseline 严格对齐)
- Stage 2 SID tensor shape `(9922, 4)`, num_emb_list=[64,128,256,1]
- Stage 4 eval 配置 (Recall@5/10/20, NDCG@5/10/20, beam_size=20, seed=42)

**Stage 0 (前置, CPU)**: 共现图重建

```bash
cd /home/wlia0047/ar57/wenyu/GeneRec
python3 scripts/task180_build_G2_cooccurrence.py \
  --train_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/train.parquet \
  --window 5 \
  --output_dir /home/wlia0047/ar57/wenyu/GeneRec/products/task180/graph/
```

输出: `products/task180/graph/G2_cooccurrence_kg_final.txt` + `kg_item_id_to_emb_id.parquet`

**Stage 1 启动命令** (Graph-aware HRQ-VAE):
```bash
cd /home/wlia0047/ar57/wenyu/GeneRec
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task180
nohup python3 scripts/task180_graph_aware_stage1_train.py \
  --lr 1e-3 \
  --epochs 200 \
  --batch_size 1024 \
  --num_workers 4 \
  --learner AdamW \
  --lr_scheduler_type linear \
  --warmup_epochs 20 \
  --data_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet \
  --graph_path /home/wlia0047/ar57/wenyu/GeneRec/products/task180/graph/G2_cooccurrence_kg_final.txt \
  --item_id_mapping /home/wlia0047/ar57/wenyu/GeneRec/products/task180/graph/kg_item_id_to_emb_id.parquet \
  --weight_decay 0 \
  --dropout_prob 0.0 \
  --bn False \
  --loss_type poincare \
  --kmeans_init True \
  --kmeans_iters 1000 \
  --sk_epsilons 0.003 0.003 0.003 \
  --sk_iters 50 \
  --device cuda:2 \
  --num_emb_list 64 128 256 \
  --e_dim 32 \
  --quant_loss_weight 1.0 \
  --beta 0.5 \
  --layers 512 256 128 64 \
  --save_limit 5 \
  --ckpt_dir /home/wlia0047/ar57/wenyu/GeneRec/products/task180/hrqvae_graph_aware/jul-25-2026_XX-XX-XX \
  > logs/task180_stage1.out 2>&1 &
```

**Stage 2 启动命令**:
```bash
python3 scripts/task180_graph_aware_stage2_codebook.py \
  --ckpt_path /home/wlia0047/ar57/wenyu/GeneRec/products/task180/hrqvae_graph_aware/<TS>/HRQVAE_best.pth \
  --output_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_graph_aware.npy \
  --device cuda:2
```

**Stage 3 启动命令** (标准 T5-small 5.5M):
```bash
cd /home/wlia0047/ar57/wenyu/GeneRec
nohup python3 scripts/task84_hgrec_stage3_train.py \
  --dataset_name Instruments \
  --dataset_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/ \
  --code_path _t5_rqvae_graph_aware.npy \
  --codebook_size 64 128 256 1 \
  --num_epochs 200 \
  --batch_size 256 \
  --lr 1e-4 \
  --num_layers 6 --num_decoder_layers 4 \
  --d_model 128 --d_ff 1024 --num_heads 6 --d_kv 64 \
  --vocab_size 1025 --max_len 20 \
  --pad_token_id 0 --eos_token_id 0 \
  --device cuda:2 \
  --mode train \
  --save_path /home/wlia0047/ar57/wenyu/GeneRec/products/task180/t5small_graph_aware/<TS> \
  --log_path /home/wlia0047/ar57/wenyu/GeneRec/logs/task180 \
  --seed 42 --early_stop 20 --beam_size 20 --infer_size 96 \
  > logs/task180_stage3.out 2>&1 &
```

**Stage 4 eval 命令**:
```bash
bash scripts/task180_graph_aware_stage4_eval.sh
```

## 3. 决策触发 (vs Task #178 新 baseline)

| 测得 R@10 | 与新 baseline 比较 | 解读 + 下一步 |
|-----------|-------------------|-------------|
| **> 新 baseline + 0.005** | 显著超过 | 共现图结构对 Stage 1 编码有正向价值, 后续可做 Step 2 (graph-aware embedding + κ-Stereographic quantizer 联合). |
| **±0.005 内** | 中性 | LightGCN 1 层不构成 semantic 加成, 不再做 Step 2. |
| **< 新 baseline - 0.005** | 显著恶化 | LightGCN 无监督平滑 label 太狠, graph-aware 路径整体不成立. |

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Stage 0 G2 共现图重建 (CPU) | ~10-15 min |
| Stage 1 Graph-aware HRQ-VAE 训练 (200 ep) | ~30-60 min (GPU 2) |
| Stage 2 Codebook 推断 | ~5-10 min (GPU 2) |
| Stage 3 T5-small 5.5M 训练 (200 ep + early_stop=20) | ~2-4 h (GPU 2) |
| Stage 4 test eval | ~2-3 min (GPU 2) |
| **总计** | **~3-5 h** |

## 5. 风险与缓解

**风险 1**: G2 共现图重建后节点数 ≠ 9922 (item id mapping drift)
**缓解**: Stage 1 入口做 sanity check assert 节点数 ≤ 9922, 不在 vocab 中的 item 自动 mask 掉

**风险 2**: LightGCN 1 层传播 + sigmoid 邻接矩阵归一化后, item_emb 范数爆掉 → HRQ-VAE encoder 数值不稳定
**缓解**: 用 row + col L2 归一化 (`D^{-1/2} A D^{-1/2}`), n_layers=1 (LightGCN 等价于 X + AX 后 mean, 不深度叠加), 输出再 unit-norm normalize

**风险 3**: Stage 1 LightGCN 传播后 Stage 2 unique SID 跌到 8000 以下 → 后续 Stage 3 严重欠拟合
**缓解**: 限制 Stage 1 LightGCN 只 1 层, 把信息含量控制在 minor perturbation 而非 dominant factor

## 6. 完成度跟踪

- [ ] G2 共现图重建 (Task #69 build_graphs.py reuse)
- [ ] Graph-aware Stage 1 launcher + LightGCN 传播层
- [ ] Stage 1 finish (best_loss_model.pth 落盘)
- [ ] Stage 2 codebook 推断 (unique SID + 利用率记录)
- [ ] Stage 3 标准 T5 训练 (复用 task178 stage3 pattern)
- [ ] Stage 4 launch + finish
- [ ] verdict 写完 (`verdicts/task180_graph_aware_result.md`)
- [ ] loop.md §16 cleanup (R8)

## 7. R-RQ-VAE framework 集成

### R7 GPU 隔离
- Phase 1 (#178) 占 GPU 0
- Phase 2 (#179) 占 GPU 1
- Phase 3 (#180) 占 GPU 2 (跟 Phase 1/2 完全并行)

### R8 §16 cleanup
- #176/#177 stale rows 已清
- #178/#179/#180 完成 → 立即删除活跃行

### R9 编号连续
- descriptions max = 178 → 新编号 179 → 新编号 180 ✅ 已 pre-creation check

### R10 主动推进
- Phase 3 Stage 0 G2 重建立即启动 (CPU only, 不抢 GPU)
- Phase 3 Stage 1 等 #178 Stage 3 出 R@10 baseline 后再启动 (避免抢占 GPU 0)

### R11 自主决策
- Stage 0 window=5 (Task #69 builder 默认, 备选 3 / 10, 不选)
- Stage 1 LightGCN n_layers=1 (备选 2/3, 不选 — 单层避免过度平滑)
- Stage 1 graph-aware encoder 维持修正版 HRQ-VAE (β=0.5, [64,128,256], poincare loss)

### R12 强制存 ckpt
- Stage 1 trainer 默认 best_loss_model.pth
- Stage 3 沿用 task84_hgrec_stage3_train.py (R12 验收过)

### R13 不用 worktree
- 全部修改直接落在共享 checkout

## 8. 关键技术细节

### LightGCN 邻居传播 (n_layers=1)

```python
import torch
import torch.nn.functional as F

def lightgcn_propagate(X, adj_norm, n_layers=1):
    """
    X: (N, D) item_emb
    adj_norm: (N, N) sparse COO 已 D^{-1/2} A D^{-1/2} 归一化
    return: (N, D) X + AX
    """
    out = X
    for _ in range(n_layers):
        out = torch.sparse.mm(adj_norm, out)
    return (X + out) / (n_layers + 1)
```

### Stage 1 LightGCN-augmented HRQ-VAE

```python
# 训练入口:
X = pd.read_parquet(item_emb_path)['embedding'].values  # (9922, 768)
X = torch.tensor(np.stack(X)).float()  # (N, D)

# G2 加载
adj = load_G2_to_sparse(graph_path, item_id_mapping, N=9922)  # sparse (N, N)
adj_norm = normalize_adj(adj)  # D^{-1/2} A D^{-1/2}

# LightGCN 1 层传播
X_graph = lightgcn_propagate(X.to(device), adj_norm.to(device), n_layers=1)

# 喂给 HRQ-VAE encoder (Phase 0 修复版)
hrqvae = HRQVAE(...)  # 修正版
recon, rq_loss, indices = hrqvae(X_graph)
```

### 共现图重建 (Task #69 build_graphs.py G2)

复用现有 `task_artifacts/scripts/task69_build_graphs.py:build_G2(src, dst, train_data, window=5)`. 仅需传入当前 Instruments `train.parquet` 即可获得 5-core 后 9922 item 节点对应的共现图.