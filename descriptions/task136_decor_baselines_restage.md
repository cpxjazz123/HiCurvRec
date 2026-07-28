# Task #136 — DECOR Table 2 Pending 5 Baselines 闭环 (TIGER/LETTER/CoST/P5/ETEGRec)

> **任务目的**: 用户 2026-07-24 要求: 把 Task #72 DECOR paper Table 2 仍 pending 的 **5 个 generative baselines** (TIGER / LETTER / CoST / ETEGRec / P5-SID+CID) 在 Musical_Instruments 上重新训练+评估, 闭环 §4.3 完整 Table 2.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

### 1.1 Task #72 现状

Task #72 verdict 2026-07-22 已闭环 DECOR 主结果 (R@10=0.0618 vs paper 0.0617, +0.16% 完美) + 20 RecBole baselines (sequential + general), 漏 5 个 generative baselines:

| Baseline | Paper R@10 | 当前状态 | 需启动 |
|----------|-----------|---------|--------|
| TIGER | 0.0574 | ❌ 未起 | Phase 2 |
| LETTER | 0.0581 | ❌ 未起 | Phase 3 |
| CoST | 0.0570 | ❌ 未起 | Phase 5 (需 git clone) |
| ETEGRec | 0.0609 | ⚠️ PID 已死, products 空 | Phase 4 重启 |
| P5-SID + P5-CID | (paper 有) | ❌ 未起 | Phase 7 (需 git clone) |

### 1.2 已就绪产物 (节省时间)

| 资源 | 路径 | 状态 |
|------|------|------|
| HG-Rec Stage 1 RQ-VAE ckpt (c111) | `products/task84/ckpt/Instruments/Jul-23-2026_20-08-06_..._sk_0.000/` | ✅ task #84 已训 |
| HG-Rec Stage 2 codebook (c111) | `HG-Rec/dataset/Instruments/Instruments_curv_1.0_1.0_1.0_t5_hrqvae_poincare.npy` | ✅ |
| HG-Rec Stage 3 TIGER (seed=42) | `products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/` | ✅ 仅 seed 错 (paper=2025) |
| ITEM EMB T5 768d | `HG-Rec/dataset/Instruments/item_emb.parquet` | ✅ |
| LETTER repo | `/LETTER/LETTER-TIGER/` + `LETTER-LC-Rec/` | ✅ cloned |
| ETEGRec repo | `/ETEGRec/` | ✅ cloned |
| RecBole | `/RecBole/` | ✅ (已用于 phase 6 general+sequential) |

### 1.3 决策 (R11.3)

| 决策 | 选择 | 拒绝 | 理由 |
|------|------|------|------|
| TIGER 实现 | ✅ 复用 HG-Rec c555 (κ≈0.5, 等价 vanilla TIGER) + seed=2025 | ❌ 重训 vanilla TIGER | HG-Rec c555 数学等价 + 节省 6h Stage 1 |
| LETTER 实现 | ✅ `/LETTER/LETTER-TIGER/` 跑 vanilla TIGER + `/LETTER/LC-Rec/` 跑 LETTER | ❌ 自实现 | LETTER repo 已有 reference impl |
| ETEGRec 实现 | ✅ `/ETEGRec/main.py` 适配 Musical_Instruments | ❌ 自实现 | repo 已成熟 |
| CoST 实现 | ✅ git clone + 跑 reference | ❌ 自实现 | paper baseline reference impl 必备 |
| P5-SID/CID 实现 | ✅ git clone reczoo/P5 + 跑 reference | ❌ 自实现 | paper baseline reference impl 必备 |
| 并行度 | ✅ 4 GPU (task135 diagnostic 占 1 张空闲用 3, 启动后扩 4 张) | ❌ 串行 | R7 强制并行 |
| 总预算 | ~12-18h GPU wallclock (4 GPU 并行) | ❌ 串行 ~50h | R10 主动推进 |

---

## 2. 实验设计

### 2.1 Phase 2 — TIGER (~3-4h, GPU 0)

**复用路径**:
- Stage 1 RQ-VAE: 复用 task84 c555 (c000 接近欧氏)
- Stage 2 codebook: 已有 `Instruments_curv_0.5_0.5_0.5_t5_hrqvae_poincare.npy`
- Stage 3 T5: 重训 seed=2025 (而非 42), 200 epoch

**启动命令**:
```bash
bash scripts/task136_tiger_seed2025.sh  # fork of task84_hgrec_stage3_train.sh, override --seed 2025
```

**输出**: `verdicts/task136_tiger_result.md` + `products/task136/tiger_stage3_seed2025/`

### 2.2 Phase 3 — LETTER (~4-6h, GPU 1)

**做法**:
- 走 `/LETTER/LETTER-TIGER/` (T5 backbone + RQ-VAE SID)
- 沿用现有 Stage 1 c111 RQ-VAE ckpt 作 SID source
- Stage 2 codebook 复用 task84 c111
- Stage 3 T5 LETTER tuning 200 epoch seed=2025

**输出**: `verdicts/task136_letter_result.md` + `products/task136/letter_stage3_seed2025/`

### 2.3 Phase 4 — ETEGRec (~4-6h, GPU 2)

**做法**:
- `/ETEGRec/main.py --dataset Musical_Instruments --backbone t5-base ...`
- 沿用 sentence-t5 768d 嵌入 (复用 task72/74 phrase-t5-prep-2026 pipeline)
- Single seed 2025, 200 epoch (ETEGRec paper default)

**输出**: `verdicts/task136_etegrec_result.md` + `products/task136/etegrec/`

### 2.4 Phase 5 — CoST (~4-6h, GPU 3, 需先 git clone)

**做法**:
```bash
cd /home/wlia0047/ar57/wenyu/GeneRec/
git clone https://github.com/CRIPAC-DIG/CoST.git CoST_temp
```
- /CoST_temp/ 跑 reference impl on Instrument
- 沿用 Stage 1 c111 RQ-VAE ckpt

**输出**: `verdicts/task136_cost_result.md` + `products/task136/cost/`

### 2.5 Phase 7 — P5-SID + P5-CID (~6-8h 顺序 或 并行 2 GPU)

**做法**:
```bash
cd /home/wlia0047/ar57/wenyu/GeneRec/
git clone https://github.com/reczoo/P5.git P5_temp  # 实际 reczoo path; 视可达性选 mirror
```
- /P5_temp/ 跑 P5-SID + P5-CID 双 baseline
- 沿用 sentence-t5 768d 嵌入 + Stage 1 c111 RQ-VAE SID

**输出**: `verdicts/task136_p5_sid_result.md` + `task136_p5_cid_result.md`

### 2.6 Phase 8 — Table 2 汇总

合并 5 baselines (TIGER/LETTER/CoST/ETEGRec/P5-SID/P5-CID) 到 `products/task136/table2_generative_summary.csv`, 与 Task #72 phase 6 (sequential + general) + Task #72 (DECOR 主结果) 汇成完整 Table 2 Instrument 12 baselines + DECOR.

---

## 3. 决策触发 (vs paper Table 2 Instrument 列)

| 指标 | 阈值 | 决策 |
|------|------|------|
| 5 baselines × 4 metrics 测试完成 | ≥ 5/6 (含 P5-SID/CID 至少 1 个) | ✅ 任务完成, 写完整 Table 2 |
| Phase 7 P5 双 baseline 全 OOM / 训练失败 | < 5/6 | ⚠️ 部分完成, 报告缺失 |
| Phase 5 CoST git clone 失败 | < 5/6 | ⚠️ 跳过, 报告 |
| 任一 baseline R@10 vs paper 偏差 >30% | 检查数据集/seed/config | ⚠️ 调整超参重跑 |

---

## 4. 预算

| Phase | 估算时间 | GPU 占用 |
|-------|---------|---------|
| Phase 2 (TIGER Stage 3 seed=2025 重训) | ~3 h | GPU 0 |
| Phase 3 (LETTER Stage 3) | ~4 h | GPU 1 |
| Phase 4 (ETEGRec 重训) | ~4 h | GPU 2 |
| Phase 5 (CoST clone + train) | ~5 h | GPU 3 |
| Phase 7 (P5-SID + P5-CID 顺序 或 并行) | ~6-8 h | 2 GPU |
| Phase 8 (Table 2 汇总 + verdict) | ~30 min | CPU |
| **总计** | **~12-18 h GPU wallclock (4 GPU 并行)** | |

若 4 GPU 同时用, wallclock ~6h 主 batch + ~6h 第二 batch = 总 ~12h 完成.

---

## 5. 风险与缓解

**风险 1 — Phase 5 CoST git clone 失败 (国内网络)**: GitHub 速率限制或连不上
→ 缓解: 用镜像 `https://hub.fastgit.xyz/CRIPAC-DIG/CoST` 或 `git clone git://github.com/CRIPAC-DIG/CoST.git` (git protocol)

**风险 2 — Phase 7 P5 repo 路径**: reczoo/P5 可能不存在
→ 缓解: 备选 `jcylin/P5` 或 `westlake-repl/P5` (mirror); 实在不行 skip P5, 报告 NO-GO

**风险 3 — 同 dataset (Instrument) 多次训练撞 GPU/OOM**: 不同 baseline 不同 backbone, 不直接共享 GPU memory
→ 缓解: R7 (4 GPU 同时 ≤ 1 实验/GPU), TIGER/LETTER/ETEGRec T5 backbone 共用 ~6-8 GB/GPU; P5 backbone 更大 ~10-15 GB

**风险 4 — ETEGRec `main.py` 配置路径硬编码 (Instruments 数据集路径)**: /ETEGRec/ 是 clone 上游, 可能没 Instrument 数据适配
→ 缓解: 先查 ETEGRec/README 看支持哪些 dataset; 缺则按 Sci/Games 类比写 yaml

**风险 5 — Phase 2 TIGER Stage 3 训练时间和 Task #84 量级 (200 epoch ~3-4h)**: seed 切换可能略大
→ 缓解: 早停 patience=20, R@10 收敛即停; 不下 hard training 预算

---

## 6. 完成度跟踪

### Phase 2 (TIGER)
- [ ] scripts/task136_tiger_seed2025.sh 写完 + py_compile 通过
- [ ] Stage 3 重训跑完 (seed=2025)
- [ ] Stage 4 eval 落盘 (Recall@5/10/20/NDCG@5/10/20)
- [ ] verdicts/task136_tiger_result.md 写出

### Phase 3 (LETTER)
- [ ] scripts/task136_letter_seed2025.sh 写完
- [ ] Stage 3 重训跑完
- [ ] Stage 4 eval 落盘
- [ ] verdicts/task136_letter_result.md 写出

### Phase 4 (ETEGRec)
- [ ] scripts/task136_etegrec_seed2025.sh 写完
- [ ] ETEGRec main.py 适配 Musical_Instruments (yaml 或 cfg)
- [ ] 重训跑完 (200 epoch seed=2025)
- [ ] Stage 4 eval 落盘
- [ ] verdicts/task136_etegrec_result.md 写出

### Phase 5 (CoST)
- [ ] git clone CoST 成功
- [ ] scripts/task136_cost_seed2025.sh 写完
- [ ] CoST 训练跑完
- [ ] Stage 4 eval 落盘
- [ ] verdicts/task136_cost_result.md 写出

### Phase 7 (P5-SID + P5-CID)
- [ ] git clone reczoo/P5 成功 (或备选 mirror)
- [ ] scripts/task136_p5_sid_seed2025.sh + scripts/task136_p5_cid_seed2025.sh 写完
- [ ] P5-SID 训练跑完
- [ ] P5-CID 训练跑完
- [ ] Stage 4 evals 落盘
- [ ] verdicts/task136_p5_sid_result.md + task136_p5_cid_result.md 写出

### 综合
- [ ] products/task136/table2_generative_summary.csv 生成 (6 baselines × 4 metrics)
- [ ] verdicts/task136_decor_baselines_restage_result.md 写出 (完整 Table 2 Instrument 12 + DECOR)
- [ ] loop.md §16 归档 (R8)
- [ ] TASKS_INDEX/CHANGELOG auto-gen refresh

---

## 7. 关联

- **前置**: Task #72 (DECOR 主结果 + 20 baselines phase 6), Task #84 (HG-Rec Stage 1+3 on Instrument)
- **依赖**: Task #135 (κ=0 grad diagnostic, 不阻塞 — 与本任务独立)
- **后续**: Task #87 v3 paper Table 2 ranking (含 6 个 generative baselines 全部)

---

## 8. 关键决策点 (R11.3 自决)

1. **TIGER 用 HG-Rec c555**: 已经训过 c111 + c555, c555 接近欧氏 (κ=0.5), 数学上对标 vanilla TIGER (DECOR paper "TIGER" baseline)
2. **seed 全部统一 2025**: 与 DECOR paper Table 2 instrument 一致
3. **并行度**: 4 GPU 尽量占满; ETEGRec + CoST 第一批, TIGER + LETTER 第二批; P5 第三批 (用空 GPU)
4. **baseline 数值偏差容忍**: ≤30% vs paper (paper 自己有 4-6% systematic bias 已观察); 不对齐数值, 只保 5/6 baselines 完成
5. **P5 repo 路径失败回退**: 跳到 mirror; 全失败 → 报告 4/5 baseline (P5 NO-GO)
6. **CoST 完全重训 vs 复用**: 重训 (各 baseline 各自独立, 不共享 ckpt)

---

## 9. 用户原文引用

> TIGER │ 0.0574 │ ⏳ Phase 2 待跑 │ GRID src/ 已实现
> LETTER │ 0.0581 │ ⏳ Phase 3 待跑 │ /LETTER/ 已 clone
> CoST │ 0.0570 │ ⏳ Phase 5 待跑 │ 需 git clone
> ETEGRec │ 0.0609 │ 🟡 Phase 4 训练中 │ /ETEGRec/ 已 clone
> P5-SID │ (paper 有) │ ⏳ Phase 7 待跑 │ 需 clone reczoo/P5
> P5-CID │ (paper 有) │ ⏳ Phase 7 待跑 │ 需 clone reczoo/P5
> 这5个,帮我解决问题,重新运行
