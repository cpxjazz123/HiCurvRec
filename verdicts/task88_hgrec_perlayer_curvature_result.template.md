# Task #88 result — HG-Rec per-layer curvature 网格搜索 (Idea1 验证)

> **完成日期**: (in progress)
> **状态**: 🟡 4/6 Stage 3 running, 2/6 pending

---

## 1. 任务目的

把 HG-Rec 硬编码全局曲率 `c=1` 改为三层独立 `c_0/c_1/c_2`. 通过网格搜索 6 个候选组合, 验证 "per-layer curvature vs single global c" 的下游 R@10 改进.

---

## 2. 实验结果

| Curvature (c_0,c_1,c_2) | Test R@5 | Test R@10 | Test N@5 | Test N@10 | Best epoch | ckpt path |
|--------------------------|----------|-----------|----------|-----------|------------|-----------|
| [1.0, 1.0, 1.0] 对照 (Task #84 baseline) | (Task #84=0.0496) | (Task #84=0.1020) | (Task #84=0.0328) | (Task #84=0.0426) | (Task #84=?) | products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/HG_Rec_best.pth |
| [0.5, 0.5, 0.5] 强双曲三层 | (TBD) | (TBD) | (TBD) | (TBD) | (TBD) | (TBD) |
| [2.0, 2.0, 2.0] 弱双曲三层 | (TBD) | (TBD) | (TBD) | (TBD) | (TBD) | (TBD) |
| [0.5, 1.0, 2.0] 逐层递减 | (TBD) | (TBD) | (TBD) | (TBD) | (TBD) | (TBD) |
| [2.0, 1.0, 0.5] 反向 | (TBD) | (TBD) | (TBD) | (TBD) | (TBD) | (TBD) |
| [1.0, 0.5, 0.5] 中→强 | (TBD) | (TBD) | (TBD) | (TBD) | (TBD) | (TBD) |

---

## 3. 决策触发 (vs Task #84 baseline R@10=0.1020)

| 网格最优 R@10 | 解读 | 决策 |
|--------------|------|------|
| R@10 > 0.115 (+12.7%) | Idea1 强证实 | 升级路径 2: c_0/c_1/c_2 设为可学习 nn.Parameter + RiemannianAdam |
| 0.105 ≤ R@10 ≤ 0.115 | Idea1 部分证实 | 路径 2 仍可选, ROI 中等 |
| R@10 < 0.105 | Idea1 失败 | 停止路径 2, 写 "per-layer 在 HG-Rec 上无显著增益" verdict |
| 网格中 c 全等优于 (c_0,c_1,c_2) 异构 | 退化证据 | 升级路径 2 风险高, 直接 NO-GO |

---

## 4. 关键产物

- `products/task88/train/curv_*` × 6 grid HRQ-VAE ckpts
- `products/task88/ckpt_hgrec/Instruments/curv_*` × 6 T5 ckpts (Stage 3)
- `dataset/Instruments/Instruments_curv_X_Y_Z_t5_hrqvae_poincare.npy` × 6 codebooks (Stage 2)
- `results/task88_eval_*.json` × 6 Stage 4 eval results (TBD)
- `products/task88/patch_notes.md` — Patch + smoke test + dry-run doc
- `scripts/task88_stage2_codebook.py` — Stage 2 codebook generator (parametric ckpt/output)
- `scripts/task88_stage4_eval.py` — Stage 4 standalone eval (loads state_dict + codebook)

---

## 5. 关键决策点 (R11.3 自决)

1. 路径选择: 用户已回答 → 路径 1 网格搜索 → 验证后再升级路径 2 ✅
2. 网格候选值: 自主选 6 组 (含 [1,1,1] 对照组)
3. Stage 1 编码复用 Task #84: 不重跑
4. Stage 3 (T5) seed=42 fixed
5. 网格全部跑, 不只看最优 (R2 禁跳过)
6. argparse `type=bool` 坑修复: 不传 `--bn`, 文档化在 patch_notes §3
7. Stage 2 codebook 第 4 列 + dedup (Task #84 fork convention): scripts/task88_stage2_codebook.py 同步

---

result: Task #88 — per-layer curvature 网格搜索 (在写)