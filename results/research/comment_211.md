## 完成：参照基线改为 F0，双列差值成为强制口径（不训练、不派发、不改历史）

### 落地物

1. **`results/research/baseline.md`**：F0 的路径、数值、SID sha256、配方与"自此为主参照"的声明。
   - `test_recall@10 = **0.057278155956754125**`（`n_eval = 57,439`，来自 `results/stage3_T5Train/TIGER_mixed_rq_F0/logs/Amazon_2023_Instruments/Nov-09-2026_08-07-21/test_final.json`）；
   - SID `results/stage2_RQ-VAE/TIGER_mixed_rq_F0/item_sids.json`，sha256 **`afa89a10b1b197a7b56da4ad322c6ddd96a3c26c04238af326d88747378bb848`**；
   - 由该 SID **复算**的结构：L1 使用码 **79**、二级前缀 **4,151**、不同三码 **18,264**（24,587 商品）；
   - 配方：E16×H16、3×256、α=0.5、3 token、Sinkhorn、`curvature=1`、3000 epoch、seed 42、全局 batch 4096、**无结构项/学习曲率/成对项/蒸馏项/径向变换**；
   - 历史对照列保留 `TIGER_mixed_rq = 0.056216`，并重述 `#198` 的教训（不得用 E32 的 SID 侧数字与 E16×H16 族比较）。
2. **`results/research/preregistration_template.md`**：参照行 `F0 = 0.057278156`（`grep` 命中），起点一律 F0，须先说明"不属已关闭三接口"，收线 `≤ 0.057278156`，塌缩闸门 `collision ≥ 0.40` 即停，**报告必须同时给出 `Δ(F0)` 与 `Δ(历史基线)` 两列**。
3. **历史零改动**：`git diff --stat` 仅显示两个新增文件（77 行），`results/stage2_RQ-VAE/**` 与 `results/stage3_T5Train/**` 下**无任何文件被改动**。

### 边界说明

按本 Issue 的边界，若你希望继续以 `TIGER_mixed_rq` 为基线（严格可比历史臂），可令本改动回退并回到 `#210` 的 B/C 决策；当前按正文要求以 F0 为主参照、历史基线保留为对照列。

**依据 commit**：`1d0bce58c`。
