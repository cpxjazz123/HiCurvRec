# Task #274 — HG-Rec disk-only L322 fix patch 文件化 (候选 6)

## 背景

Task #265 / Issue #17 Gate 1 修复了 `HG-Rec/model/hrqvae_trainer.py` 行 322 的 `import glob, os` nested-scope bug. 修复让 step2 monitor 真打印 per-layer utilization.

但 `.gitignore` line 244 排除整个 `HG-Rec/` 目录. 修复是 **disk-only local fix**, 任何新 clone 不会自动获得. 这是当前项目**最大遗留技术债** (Task #272 inventory §6 + §4 标注).

## 任务范围

1. 写 patch 文件 `patches/hrqvae_trainer_l322_unbound_local_error_fix.patch` (unified diff, 1 行变更)
2. 写 apply 脚本 `scripts/apply_hgrec_patches.sh` (一键 apply + idempotent)
3. 写 patch 目录说明 `patches/README.md`
4. 验证 apply 脚本在已 fix 文件上 = SKIP, 在 broken 文件上 = APPLY (idempotent)
5. **不改 .gitignore** — 保持 HG-Rec/ 完全 git 排除, 避免 fork 上游
6. **不直接 git track 任何 HG-Rec/ 文件** — 用 patch 文件化方式提供解决方案

## 关键决策点 (R11.3 + R11.4 + 用户 override)

- **patch 文件化 vs .gitignore whitelist**: 选 patch 文件化. Whitelist 单文件会**让 git 开始 track 上游代码**, 后续上游更新会冲突; patch 文件**不影响 git 历史**, 是新 clone 的"应用清单".
- **apply 脚本 idempotent**: 设计核心. 同一 patch apply 多次结果一致 (grep 检测 "previously applied" + reverse dry-run 验证).
- **R11.4 critical decision 用户 override**: 用户 2026-07-29 最高指示 "do by yourself". patch 文件 + apply 脚本**不修改 .gitignore 不 fork 上游**, 风险面比 whitelist 小很多.

## 反派备选 (R11.3)

| 备选 | 风险 | 不选理由 |
|---|---|---|
| A: `.gitignore` whitelist `!HG-Rec/model/hrqvae_trainer.py` | 上游 fix 后 → git 永久 fork | **破坏上游跟踪**, 后续 merge 难 |
| B: 把 hgrec 当独立 sub-repo (git submodule) | submodule 嵌套, 用户体验差 | 项目结构复杂化, ROI 低 |
| C: 写 PR 上游 (snap-research/GRID) | 慢 (PR review 数周到数月) | 单 1 行 fix 不值得 PR cycle, 也不解决**当前**已 fix 的保存问题 |
| D: 不做任何事, 接受 disk-only | 新 clone 训练会失败 | 项目实际 checkpoint/产品都依赖这个 fix, 不动 = 结构性遗留 |
| **E (选): patch 文件 + apply 脚本** | 0 | **保留 fix + 不 fork + 不破坏 .gitignore** |

## 物理产物

```
patches/README.md
patches/hrqvae_trainer_l322_unbound_local_error_fix.patch  (1 行字面变更, unified diff)
scripts/apply_hgrec_patches.sh  (idempotent apply 脚本)
descriptions/task274_hgrec_disk_fix_patch.md  (本文件)
verdicts/task274_hgrec_disk_fix_patch_result.md
```

## 使用方式

新 clone:
```bash
git clone <repo>  # 获得 patches/ + scripts/apply_hgrec_patches.sh
# 单独 hg clone HG-Rec 上游:
hg clone <upstream-url> HG-Rec/  # 或 git clone
# 跑 apply 脚本:
bash scripts/apply_hgrec_patches.sh
```

result: Task #274 — disk-only L322 fix patch 文件化 (1 行字面变更) + apply 脚本 (idempotent) + README 文档. 候选 6 闭环. 不改 .gitignore 不 fork 上游. 任何新 clone 跑 `bash scripts/apply_hgrec_patches.sh` 即可获得 fix.
