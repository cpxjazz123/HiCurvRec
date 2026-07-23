# Task #72 — DECOR 主结果复现 (Musical_Instruments, Table 2 全 12 baselines + DECOR)

> **任务目的**: 在 Amazon Musical_Instruments 2023 数据集上**完整复现 DECOR paper §4.3 Table 2 Instrument 列** — DECOR + 全部 12 baselines (Caser/GRU4Rec/SASRec/BERT4Rec/FDSA/S³Rec/P5-SID/P5-CID/TIGER/LETTER/CoST/ETEGRec) × 4 metrics (R@5/R@10/N@5/N@10) × 单 seed 2025 × 200 epoch.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

### 1.1 DECOR 论文核心
DECOR (SIGIR '26, arXiv: yliuaa/DECOR) 在 Amazon Reviews 2023 三个子集 (Scientific / **Instrument** / Game) 上系统对比 12 个 baseline + DECOR 自身, 报告 4 个 top-K 指标. Instrument 列的对照目标是:

| Method | R@5 | R@10 | N@5 | N@10 |
|--------|-----|------|-----|------|
| TIGER | 0.0368 | 0.0574 | 0.0242 | 0.0308 |
| LETTER | 0.0372 | 0.0581 | 0.0243 | 0.0310 |
| CoST | 0.0366 | 0.0570 | 0.0242 | 0.0306 |
| ETEGRec | 0.0387 | 0.0609 | 0.0251 | 0.0323 |
| **DECOR** | **0.0409** | **0.0617** | **0.0272** | **0.0339** |

DECOR 在 Instrument 上 R@5/R@10/N@5/N@10 比最强 baseline ETEGRec 高 **+5.7% / +1.3% / +8.4% / +5.0%**, NDCG@10 提升最显著.

### 1.2 现状
- DECOR repo 已 clone: `/home/wlia0047/ar57/wenyu/GeneRec/DECOR/` (449 MB, yliuaa/DECOR)
- 已转 md: `papers/DECOR.md` (76 KB, 399 lines)
- 已 clone 对照 repo: `GeneRec/LETTER/` (HonghuiBao2000/LETTER), `GeneRec/ETEGRec/` (BishopLiu/ETEGRec), `GeneRec/RecBole/` (99 算法 framework)
- DECOR `cache/AmazonReviews2023/` 只有 **Industrial_and_Scientific** (274 MB, 含 5-core arrow + sentence-t5 嵌入 + RQ-VAE SID 产物), **缺 Musical_Instruments**
- DECOR paper 训练用 1×A40 GPU, seed=2025, 200 epoch default

### 1.3 用户决策 (2026-07-21 AskUserQuestion)
1. **范围**: DECOR + 全部 12 baselines (Table 2 完整复现)
2. **数据集**: Musical_Instruments (DECOR paper 中标记为 "Instrument")
3. **genrec 依赖**: 安装旧版 genrec (DECOR 期望的 API, phonism/genrec 新版 `genrec/models/` 复数 API 不兼容)
4. **训练规模**: 单 seed 2025 + 200 epoch (paper 默认)

---

## 2. 实验设计

### 2.1 变量
- **Method** (13 个: DECOR + 12 baselines)
- **共享**: 数据集 = Amazon Musical_Instruments 2023 (5-core, max seq length=20, leave-one-out 评估), seed=2025, 评估指标 R@5/R@10/N@5/N@10

### 2.2 保持不变
- 数据集: `Musical_Instruments` (McAuley-Lab/amazon-reviews-2023, 5-core)
- Backbone: Sentence-T5-base (text encoder) + T5 (recommender)
- SID: 256 codebook × 3 levels = 4-token SID (paper §4.1.2)
- 评估: leave-one-out (last = test, second-last = val), top-K full ranking
- 训练规模: 单 seed 2025, 200 epoch (paper 默认)
- DECOR 超参: α=0.35, bos_queries=64 (paper train.sh)

### 2.3 Phase 划分 (按 codebase 复用)

| Phase | Method | Codebase | 复用情况 | 启动命令骨架 |
|-------|--------|----------|---------|------------|
| **0** | Setup | - | - | 装旧版 genrec, 下载 Musical_Instruments, sentence-t5 嵌入, RQ-VAE SID |
| **1** | **DECOR** | GeneRec/DECOR/ | 主方法, 必须从头跑 | `accelerate launch DECOR/main.py --model=DECOR --dataset=AmazonReviews2023 --category=Musical_Instruments --metadata=sentence --alpha=0.35 --bos_queries=64 --run_id=Instruments` |
| **2** | TIGER | GeneRec/src/ (GRID pipeline) | GRID 已实现的 TIGER 训练 (Task #59 Toys R@5=0.0857 baseline) | `python -m src.train experiment=tiger_train_flat data_dir=data/amazon_data/musical_instruments ...` |
| **3** | LETTER | GeneRec/LETTER/ | 已 clone, 需适配 Musical_Instruments | 按 LETTER README |
| **4** | ETEGRec | GeneRec/ETEGRec/ | 已 clone, 需适配 Musical_Instruments | 按 ETEGRec README |
| **5** | CoST | 需 git clone 或实现 | 论文 baseline, 暂未 clone | (Phase 0 时一起 clone) |
| **6** | Caser + GRU4Rec + SASRec + BERT4Rec + FDSA + S³Rec | GeneRec/RecBole/ | 6 个传统 sequential baselines, RecBole 99 算法 framework 都有 | 按 RecBole 配置 |
| **7** | P5-SID + P5-CID | 需 git clone | 论文 baseline | (Phase 0 时一起 clone) |
| **8** | 汇总 | - | - | 收集 13 methods 4 metrics 到 Table 2 Instrument |

### 2.4 Phase 0 Setup 详细步骤

```bash
# 1. 装旧版 genrec (DECOR 期望的 API)
pip install genrec==<old_version>  # 需要先查 DECOR 提交历史, 找到匹配版本
# 旧 API: genrec.model.AbstractModel / genrec.tokenizer.AbstractTokenizer /
#        genrec.evaluator.Evaluator / genrec.utils.{get_file_name, get_total_steps, ...}

# 2. 下载 Musical_Instruments 数据 (~250 MB)
python -c "from datasets import load_dataset; ds = load_dataset('McAuley-Lab/amazon-reviews-2023', '5core_last_out_w_his_Musical_Instruments')"

# 3. Sentence-T5 嵌入 + RQ-VAE SID (沿用 Industrial_and_Scientific 流程)
python DECOR/main.py --dataset=AmazonReviews2023 --category=Musical_Instruments --metadata=sentence --inference=1
# 期望产出:
#   cache/AmazonReviews2023/Musical_Instruments/processed/sentence-t5-base.sent_emb
#   cache/AmazonReviews2023/Musical_Instruments/processed/sent_rqvae.pth
#   cache/AmazonReviews2023/Musical_Instruments/processed/sentence-t5-base_256,256,256,256.sem_ids
```

### 2.5 决策触发阈值 (vs paper Table 2 Instrument 列)

| 条件 | DECOR R@10 结果 | 决策 |
|------|----------------|------|
| **DECOR ≥ 0.0617** (paper 值) | ≥ 0.0617 | ✅ 完全复现, 验证 §4.3 主结果 |
| **DECOR ∈ [0.0574, 0.0617)** | 训练流程跑通, 略低 | ⚠️ 复现部分成功, 检查超参 |
| **DECOR < 0.0574** (< TIGER) | 流程有问题 | ❌ 检查数据 / embedding / SID / 模型代码 |
| **DECOR 不是 12 baselines 第一** | 排名不达 | ❌ §4.3 主结论 FALSIFIED |

| 条件 | 全部 13 methods 完成度 | 决策 |
|------|------------------------|------|
| **13/13 跑完** | 完整 Table 2 | ✅ 任务完成 |
| **≥ 6/13 跑完** (含 DECOR + 3 generative) | 部分 baseline 缺失 | ⚠️ 阶段完成, 后续补充 |
| **< 6/13** | 阻塞 | ❌ 重新评估 |

---

## 3. 预算

| 阶段 | 估算时间 (单 A40) |
|------|-------------------|
| Phase 0 (genrec 旧版安装 + Musical_Instruments 下载 + sentence-t5 嵌入 + RQ-VAE SID) | ~4-6 hours |
| Phase 1 (DECOR 训练, 200 epoch) | ~6-8 hours |
| Phase 2 (TIGER 训练) | ~3-4 hours |
| Phase 3 (LETTER 训练) | ~4-6 hours |
| Phase 4 (ETEGRec 训练) | ~4-6 hours |
| Phase 5 (CoST 训练) | ~4-6 hours |
| Phase 6 (6 traditional sequential baselines) | ~3-6 hours / each = 18-36 hours |
| Phase 7 (P5-SID + P5-CID) | ~3-4 hours / each = 6-8 hours |
| Phase 8 (汇总 Table 2) | ~1 hour |
| **总计** | **~50-80 hours (~2-3 天单卡 / ~12-20 hours 用 4 卡并行)** |

预算优先级:
1. **必须**: Phase 0 + Phase 1 (DECOR) + Phase 2 (TIGER) = ~13-18 hours, 验证 DECOR > TIGER 主结论
2. **高 ROI**: + Phase 3 (LETTER) + Phase 4 (ETEGRec) = ~22-30 hours, 验证 DECOR > all generative
3. **完整 Table 2**: + Phase 5/6/7 = ~50-80 hours

---

## 4. 风险与缓解

**风险 1**: 旧版 genrec 安装困难 (phonism/genrec 已经过 API 重构, 老版本可能没有 wheel 或已从 PyPI 移除)
→ 缓解: (a) 查 phonism/genrec git history, 找 `genrec.model` 时代的 commit, `pip install git+https://github.com/phonism/genrec.git@<commit>`; (b) 实在不行 fallback 到修复 DECOR trainer.py (但 R7 禁止 fallback, 需用户授权)

**风险 2**: 12 baselines 多个 codebase 配置混乱, 评估口径不统一 (DECOR 用 leave-one-out, 部分 baseline 用 random split)
→ 缓解: 统一用 DECOR 的 evaluator (`GeneRec/DECOR/evaluator.py`) 评估所有 13 methods; baseline 训练沿用各自论文的 split, 但评估用 DECOR 口径

**风险 3**: Musical_Instruments 数据下载慢或失败 (HF datasets 服务在某些地区可能慢)
→ 缓解: 用 `HF_ENDPOINT=https://hf-mirror.com` 镜像; 或下载 huggingface_hub 缓存到本地

**风险 4**: DECOR 200 epoch 训练时间超预算 (单 A40 可能 8-12 hours)
→ 缓解: 先跑 50 epoch sanity check (~2 hours), 验证 loss 下降, 再决定是否继续

**风险 5**: 4 卡 GPU 被 vLLM (GPU 0/1) 占用, 实际可用 2 卡 (GPU 2/3)
→ 缓解: 按 R7 规则, 新任务绑定到完全空闲 GPU (util<10%, mem<5GB), 不能抢用 vLLM 占用的卡; 串行跑或只用 2 张空闲卡

**风险 6**: P5-SID/P5-CID/CoST 没 clone, 需要先 git clone
→ 缓解: Phase 0 时一并 clone: P5 (reczoo/P5), CoST (CRIPAC-DIG/CoST)

---

## 5. 完成度跟踪

- [ ] **Phase 0**: Setup (genrec 旧版安装 + Musical_Instruments 下载 + sentence-t5 嵌入 + RQ-VAE SID)
- [ ] **Phase 1**: DECOR 训练 (200 epoch, seed 2025)
- [ ] **Phase 2**: TIGER baseline
- [ ] **Phase 3**: LETTER baseline
- [ ] **Phase 4**: ETEGRec baseline
- [ ] **Phase 5**: CoST baseline
- [ ] **Phase 6**: Caser + GRU4Rec + SASRec + BERT4Rec + FDSA + S³Rec (6 traditional)
- [ ] **Phase 7**: P5-SID + P5-CID
- [ ] **Phase 8**: 评估汇总, 生成 Table 2 Instrument
- [ ] **Phase 9**: 写 verdict (`verdicts/task72_decor_main_repro_instrument_result.md`)

---

## 6. 关联

- **DECOR paper**: `papers/DECOR.pdf` (1.2 MB) + `papers/DECOR.md` (76 KB, 已转)
- **DECOR repo**: `GeneRec/DECOR/` (449 MB, yliuaa/DECOR)
- **Baseline repos**: `GeneRec/LETTER/`, `GeneRec/ETEGRec/`, `GeneRec/RecBole/`
- **前置**: 无 (新任务)
- **后续候选**:
  - Task #73: DECOR + Scientific + Game (复现另外 2 个 dataset)
  - Task #74: DECOR §4.4 Ablation (Table 3)
  - Task #75: DECOR §4.5 Hyperparameter (Figure 3)
  - Task #76: DECOR §4.6-§4.8 case study + convergence + cost